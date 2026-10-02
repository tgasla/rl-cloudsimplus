#!/usr/bin/env python3
"""Reference policies on the RING-N members: runner, outputs and gates (refpol_spec.md §6-7).

  python3 benchmark/run_references.py --members S,PI-S --split test \\
      --policies R0,R1,R2,R3-fam --rollouts 8 --num-cpu 16 --out-dir benchmark/results --check

Each (policy, rollout) is one play_levels pass over the whole split, on one vectorised env per
member (the live gateway, one JVM per worker). R0 plays --rollouts passes, the others one.
Before any JVM starts, one multiprocessing pool (--plan-workers) computes every level's
zero-contention ceilings and, when R4 is played, its R4-CDLS plan (r4_planner, cached under
--plan-cache). R4 is the derived row: per level the best of R4-CDLS and the rule family.
Outputs in --out-dir are merged with what is already there: a re-run replaces that policy's
rows for that member and split, and the derived R4 rows are rebuilt.
  references_<split>.csv          one row per (policy, member, split, level_id, rollout)
  references_summary_<split>.csv  one row per (member, level_id): G_rand, G_ref, ...
  references_meta.json            provenance, and every default decision under "decisions"
Row columns beyond the spec's, the window-blocking measurement of spec §9.6: decisions,
window_full_steps (decision steps with all max_jobs_waiting slots taken), deferred_slots (real
slots left with the no-op) and hidden_jobs (arrived, unplaced jobs outside the window, summed
over the decision steps).

Exit status: 1 if a gate fails (--check / --check-only), 2 on a usage error.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from stable_baselines3.common.vec_env import VecEnvWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RL_MANAGER = os.path.join(REPO, "common", "rl-manager")
for path in (os.path.join(RL_MANAGER, "gym_cloudsimplus"), RL_MANAGER, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import levels  # noqa: E402
from utils.evaluation import EPISODE_SUMS, level_quotas, play_levels  # noqa: E402
from utils.misc import (_check_datacenter_amounts_are_one, _check_datacenters_unique,  # noqa: E402
                        _translate_connect_to_names_to_idx, vectorize_env)

import r4_planner  # noqa: E402
from reference_policies import (POLICIES, R0_SEED, R3_FAMILY, R4_POLICY,  # noqa: E402
                                ReferencePredictor)

TOPOLOGIES_DIR = os.path.join(REPO, "common", "topologies")
MANIFEST = os.path.join(TOPOLOGIES_DIR, "ring", "manifest.json")
DEFAULT_CONFIG = os.path.join(REPO, "domain", "job-placement", "config.yml")
DEFAULT_JAR = os.path.join(REPO, "domain", "job-placement", "cloudsimplus-gateway", "build",
                           "libs", "cloudsimplus-gateway-0.1.0.jar")

ROW_COLUMNS = [
    "policy", "member", "split", "level_id", "rollout", "source",
    "unshaped_return", "shaped_return", "steps", *EPISODE_SUMS,
    "offered_value", "terminated", "worker",
    "zc_ceiling", "zc_ideal",
    "r4_pred_return", "r4_divergent_steps", "r4_first_divergence_step",
    "elapsed_s",
    "decisions", "window_full_steps", "deferred_slots", "hidden_jobs",
]
SUMMARY_COLUMNS = ["member", "split", "level_id", "G_rand", "G_rand_sd", "G_R1", "G_R2", "G_R3",
                   "G_R4_cdls", "G_ref", "ref_source", "zc_ceiling", "rho_member"]
R4_CANDIDATES = (R4_POLICY, "R1", "R2") + R3_FAMILY    # tie order: CDLS first
RELABEL_PAIRS = (("S", "PI-S"), ("C1-N19", "PI-N19"))
G1_STRICT = ("R1", "R3")
G1_WARN = ("R0", "R2", R4_POLICY, "R4")
G2_FACTOR = 5.0
G3_MIN_GAP = 0.05                                      # mean_c(zc - G_ref) on members with rho >= 1
G3_RHO = 1 - 1e-6
MIN_HORIZON = levels.LAST_ARRIVAL + 1                  # 161

DECISIONS = {
    "horizon": "H = max_episode_length from config.yml (200); the runner refuses H < "
               "levels.HORIZON (200), the minimum utils.misc.level_stream enforces for RING-N, "
               "so spec 9.1 is resolved. The H_consistency gate keeps the spec's threshold "
               "LAST_ARRIVAL + 1 (161)",
    "gate2": "G2 fails only on the default reading mean_c(G_ref - G_rand) > 5 sd_c(G_rand) and "
             "min_c(G_ref - G_rand) > 0 (sd over the split's levels, ddof 1); the paired reading "
             "mean/sd_c(G_ref - G_rand) and the max(sd(G_ref), sd(G_rand)) reading are warnings",
    "r4_wording": "R4 = clairvoyant density list scheduler with exact simulation, guarded by "
                  "hindsight selection over the rule family (spec default)",
    "r0_noop": "R0 draws uniformly from [no-op] + the legal DCs (spec default)",
    "lock_member": "LOCK's test split counts among the 12 members; the lockbox split is refused "
                   "without --unlock-lockbox (spec default)",
    "window_blocking": "measured (window_full_steps, deferred_slots, hidden_jobs per episode) "
                       "and reported; the environment is unchanged: no reject action, no early "
                       "eviction",
    "r1": "the gateway's rule (cloudlet_to_dc_mapping earliest-shortest-to-most-free-dc), which "
          "decides on the agent's window and observation too: jobs by (ttd, r, slot); the legal "
          "DC with the most free PEs (sum of its hosts' observed free_pes, a job placed this step "
          "using its cores on the DC's most-free host, clipped at 0), then the least backlog "
          "(sum of its hosts' backlog_core_ts plus the core-timesteps placed there this step), "
          "then (tier, capacity desc, name); always binds",
    "r2": "the gateway's rule (cloudlet_to_dc_mapping earliest-most-critical-to-nearest-dc), on "
          "the same window and observation: jobs by (ttd, sensitivity desc, slot); legal DCs by "
          "hops (origin 0, ring neighbours 1, cloud 2) then (tier, capacity desc, name); the "
          "first whose most-free host has free_pes (less this step's placements) >= cores, else "
          "the no-op. Supersedes the spec's 'origin if the mask allows it'",
    "r3_min_ect0": "R3's step-start order key minECT0 is the minimum ECT over the legal DCs it "
                   "considers (cost < V + P); inf if none",
    "config_parsing": "only config.yml's common: section is read; custom YAML tags are stubbed "
                      "as common/scripts/preflight.py does, since the experiments' !include "
                      "paths resolve only inside the container",
    "r4_cdls": "spec 5B with g = 0 (the port is bit-exact) and three readings: (1) the queue "
               "check protects a committed job that is in the window without the new job "
               "('stays at rank < 32'), not one already pushed out; (2) a rejected job stays in "
               "the pool until its due (the environment has no reject action), so where that "
               "would push committed jobs out of the window, which the queue check forbids a "
               "commit, it is dumped instead if that loses less than those jobs' V + P: bound "
               "at its first visible step on the legal DC that loses the least value (its cost, "
               "less its own V + P if on time there, plus V + P of every met job it makes late); "
               "(3) in Phase B a hidden job that no DC can take on time without harm is dumped "
               "the same way instead of dropped. The no-harm check keeps every met job met. "
               "Among DCs the cheapest cost group is searched first (cost is the first choice "
               "key)",
    "zc": "zc_ceiling: per job the best of -P and, per legal DC, V - c if the port runs it on "
          "time alone on the empty DC bound at max(arrival, 1), else -P - c; zc_ideal replaces "
          "the port run with nd + mi / mips <= deadline; both / Z, Z checked against every "
          "row's offered_value",
}


# ─── Parameters ─────────────────────────────────────────────────────────────────────────

def load_common(config_path: str) -> dict:
    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    with open(config_path) as f:
        return dict(yaml.load(f, Loader=_Loader)["common"])


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def member_topology(member_id: str, manifest: dict) -> list:
    """The member's datacenters as the environment takes them (test_gateway_e2e._ring_member)."""
    entry = {m["id"]: m for m in manifest["members"]}[member_id]
    topology = levels.load_topology(os.path.join(TOPOLOGIES_DIR, entry["yaml"]))
    for dc in topology:
        dc["connect_to"] = levels._as_list(dc.get("connect_to", []))
        dc["hosts"] = levels._as_list(dc["hosts"])
        for host in dc["hosts"]:
            host["vms"] = levels._as_list(host["vms"])
    _check_datacenters_unique(topology)
    _check_datacenter_amounts_are_one(topology)
    return _translate_connect_to_names_to_idx(topology)


def member_params(common: dict, member: str, manifest: dict, split: str, num_cpu: int,
                  horizon: int, base_port: int | None) -> dict:
    params = dict(common)
    params.update(datacenters=member_topology(member, manifest), benchmark_member=member,
                  ring_manifest=MANIFEST, level_split=split, cloudlet_to_dc_mapping="rl",
                  rl_problem="job_placement", num_cpu=num_cpu, grpc_base_port=base_port,
                  log_dir=None, save_experiment=False, max_episode_length=horizon,
                  # required by the simulator's settings parser; set by entrypoint.py in a run
                  mode="evaluate", num_experiments=1)
    return params


def params_hash(params: dict) -> str:
    stable = {k: v for k, v in params.items() if k not in ("grpc_base_port", "num_cpu")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


def member_rho(manifest: dict, member_id: str, anchor: dict) -> float:
    entry = {m["id"]: m for m in manifest["members"]}[member_id]
    if "rho" in entry["load"]:
        return float(entry["load"]["rho"])
    return levels.resolve_lambda(manifest, member_id, anchor) * levels.w_bar(anchor) \
        / entry["capacity_pes"]


def free_port_block(n: int) -> int:
    """The first of n consecutive free ports, from a random start so that concurrent runs
    (and the e2e tests, which scan from 52000) rarely race for the same block."""
    starts = list(range(40000, 60000 - n, n))
    random.shuffle(starts)
    for base in starts:
        socks = []
        try:
            for port in range(base, base + n):
                s = socket.socket()
                socks.append(s)
                s.bind(("", port))
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port block")


def expand_policies(spec: str) -> list:
    out = []
    for name in (p.strip() for p in spec.split(",") if p.strip()):
        names = {"R3-fam": list(R3_FAMILY), "R4": [R4_POLICY]}.get(name, [name])
        for n in names:
            if n not in POLICIES and n != R4_POLICY:
                raise SystemExit(f"unknown policy {n!r}")
            if n not in out:
                out.append(n)
    return out


# ─── Provenance ─────────────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The code the numbers depend on, hashed per run: much of it is uncommitted, which git_sha
# cannot show.
CODE_FILES = ("benchmark/refpol_common.py", "benchmark/reference_policies.py",
              "benchmark/run_references.py", "benchmark/r4_planner.py",
              "benchmark/cloudsim_port.py", "common/rl-manager/utils/evaluation.py",
              "common/rl-manager/utils/misc.py", "common/rl-manager/utils/levels.py",
              "common/rl-manager/gym_cloudsimplus/gym_cloudsimplus/envs/job_placement.py")


def code_sha256() -> dict:
    return {path: sha256_file(os.path.join(REPO, path)) for path in CODE_FILES
            if os.path.exists(os.path.join(REPO, path))}


def git_state() -> tuple:
    def git(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


# ─── Levels: ceilings and plans ─────────────────────────────────────────────────────────

def _prepare_level(task: tuple) -> tuple:
    member, level, params, jobs_json, plan_cache = task
    zc = r4_planner.zc_ceiling(params, jobs_json)
    plan = None if plan_cache is None else \
        r4_planner.cached_plan(params, jobs_json, member, level, plan_cache)
    return member, level, zc, plan


def prepare_levels(member_params: dict, split: str, with_plans: bool, args) -> dict:
    """Per member and level of the split: the zero-contention ceilings (spec 5D) and, with
    with_plans, the R4-CDLS plan (spec 5B, cached under args.plan_cache), computed in one
    multiprocessing.Pool(args.plan_workers) before any JVM starts (the pool forks).
    Returns {member: {level: {"zc_ceiling", "zc_ideal", "Z", "plan"}}}."""
    tasks = []
    for member, params in member_params.items():
        source = levels.LevelSource(MANIFEST, member, [dc["name"] for dc in params["datacenters"]])
        tasks += [(member, level, params, source.jobs_json(level),
                   args.plan_cache if with_plans else None) for level in levels.EVAL_SPLITS[split]]
    out = {member: {} for member in member_params}
    start = time.perf_counter()
    workers = max(1, min(args.plan_workers, os.cpu_count() or 1, len(tasks)))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for member, level, (zc, ideal, Z), plan in pool.imap_unordered(_prepare_level, tasks):
            out[member][level] = {"zc_ceiling": zc, "zc_ideal": ideal, "Z": Z, "plan": plan}
    elapsed = time.perf_counter() - start
    if with_plans:
        for member, by_level in out.items():
            plans = [by_level[level]["plan"] for level in sorted(by_level)]
            took = [p["stats"]["planning_s"] for p in plans]
            print(f"  {member:7s} R4-CDLS plans: predicted G = "
                  f"{np.mean([p['return'] for p in plans]):+.4f}, planning {np.mean(took):.1f} s "
                  f"mean, {max(took):.1f} s max per level (when planned)", flush=True)
    print(f"  levels prepared in {elapsed:.1f} s ({len(tasks)} levels, {workers} processes)",
          flush=True)
    return out


# ─── Playing ────────────────────────────────────────────────────────────────────────────

class InfoTap(VecEnvWrapper):
    """Hands every step's infos to `observe`: play_levels keeps only the episode sums."""

    def __init__(self, venv, observe):
        super().__init__(venv)
        self.observe = observe

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.observe(infos)
        return obs, rewards, dones, infos


def play_policy(env, policy: str, rollout: int, split: str, num_cpu: int, strict: bool,
                **policy_kwargs) -> list:
    """Rows of one pass of `policy` over the split; checks every level was played once."""
    predictor = ReferencePredictor(env, policy, rollout, strict=strict, **policy_kwargs)
    start = time.perf_counter()
    rows = play_levels(InfoTap(env, predictor.observe), predictor, level_quotas(split, num_cpu))
    elapsed = time.perf_counter() - start
    played = sorted(int(r["level_id"]) for r in rows)
    if played != list(levels.EVAL_SPLITS[split]):
        raise RuntimeError(f"{policy} rollout {rollout} did not play every {split} level once")
    seen = {}
    for row in rows:
        w = row["worker"]
        record = predictor.episodes[w][seen.get(w, 0)]
        seen[w] = seen.get(w, 0) + 1
        if record["level_id"] != row["level_id"]:
            raise RuntimeError(f"worker {w}: episode record for level {record['level_id']}, "
                               f"row for {row['level_id']}")
        row.update({k: v for k, v in record.items() if k != "level_id"})
        row.update(policy=policy, rollout=rollout, elapsed_s=elapsed,
                   source="CDLS" if policy == R4_POLICY else policy)
    return rows


def run_member(member: str, params: dict, policies: list, args, prepared: dict) -> tuple:
    """(rows, per-pass timings) of every policy pass on one member; prepared: its levels'
    ceilings and plans (prepare_levels)."""
    split = args.split
    rows, timings = [], []
    r4_kwargs = {"plans": {level: p["plan"] for level, p in prepared.items()}}
    env = vectorize_env(None, None, num_cpu=args.num_cpu, params=params, jobs_json="[]")
    try:
        for policy in policies:
            for rollout in range(args.rollouts if policy == "R0" else 1):
                kwargs = r4_kwargs if policy == R4_POLICY else {}
                new = play_policy(env, policy, rollout, split, args.num_cpu,
                                  strict=args.strict_r4, **kwargs)
                for row in new:
                    level = prepared[row["level_id"]]
                    if row["offered_value"] != level["Z"]:
                        raise RuntimeError(f"{member} level {row['level_id']}: offered_value "
                                           f"{row['offered_value']} is not the ceiling's Z {level['Z']}")
                    row.update(member=member, split=split, zc_ceiling=level["zc_ceiling"],
                               zc_ideal=level["zc_ideal"])
                rows.extend(new)
                g = np.array([r["unshaped_return"] for r in new])
                elapsed = new[0]["elapsed_s"]
                timings.append({"member": member, "policy": policy, "rollout": rollout,
                                "elapsed_s": round(elapsed, 2)})
                print(f"  {member:7s} {policy:7s} rollout {rollout}: G = {g.mean():+.4f} "
                      f"± {g.std(ddof=1):.4f} (sd over {len(g)} levels)  per episode: "
                      f"window full {np.mean([r['window_full_steps'] for r in new]):5.1f} steps, "
                      f"deferred {np.mean([r['deferred_slots'] for r in new]):7.1f} slots, "
                      f"hidden {np.mean([r['hidden_jobs'] for r in new]):7.1f} job-steps  "
                      f"{elapsed:6.1f} s", flush=True)
    finally:
        env.close()
    return rows, timings


# ─── Tables ─────────────────────────────────────────────────────────────────────────────

def build_r4(df: pd.DataFrame, require_cdls: bool = True) -> pd.DataFrame:
    """The derived R4 rows: per (member, split, level), the best of R4-CDLS and the rule family
    (rollout 0), with `source` naming the winner (CDLS, R1, ...); ties go to the earlier
    candidate. Without R4-CDLS rows a level gets none, unless require_cdls is False (the rule
    family alone: the portfolio guard)."""
    cand = df[df["policy"].isin(R4_CANDIDATES) & (df["rollout"] == 0)]
    out = []
    for _, grp in cand.groupby(["member", "split", "level_id"], sort=True):
        if require_cdls and R4_POLICY not in set(grp["policy"]):
            continue
        grp = grp.iloc[np.argsort([R4_CANDIDATES.index(p) for p in grp["policy"]], kind="stable")]
        best = grp.iloc[int(np.argmax(grp["unshaped_return"].to_numpy()))].copy()
        best["source"] = "CDLS" if best["policy"] == R4_POLICY else best["policy"]
        best["policy"], best["rollout"] = "R4", 0
        out.append(best)
    return pd.DataFrame(out, columns=df.columns).reset_index(drop=True)


def summarise(df: pd.DataFrame, rho: dict) -> pd.DataFrame:
    """One row per (member, split, level): G_rand (mean over R0 rollouts) and its sd, the rule
    returns, R4-CDLS, G_ref (the R4 row) and its source."""
    keys = ["member", "split", "level_id"]
    base = df[keys].drop_duplicates().sort_values(keys).set_index(keys)
    by = df.groupby(["policy"] + keys)["unshaped_return"]
    mean, sd = by.mean(), by.std(ddof=1)

    def col(policy, table=mean):
        return table.loc[policy] if policy in table.index.get_level_values(0) else np.nan

    out = base.assign(G_rand=col("R0"), G_rand_sd=col("R0", sd), G_R1=col("R1"),
                      G_R2=col("R2"), G_R3=col("R3"), G_R4_cdls=col(R4_POLICY), G_ref=col("R4"))
    r4 = df[df["policy"] == "R4"].set_index(keys)["source"]
    out["ref_source"] = r4 if len(r4) else np.nan
    out["zc_ceiling"] = df.groupby(keys)["zc_ceiling"].first()
    out = out.reset_index()
    out["rho_member"] = out["member"].map(rho)
    return out[SUMMARY_COLUMNS]


# ─── Gates ──────────────────────────────────────────────────────────────────────────────

def gate_horizon(horizon: int, allow_short: bool) -> list:
    ok = horizon >= MIN_HORIZON
    status = "PASS" if ok else ("WARN" if allow_short else "FAIL")
    return [("H_consistency", status, f"H = {horizon}, needs >= {MIN_HORIZON} "
             f"(LAST_ARRIVAL + 1)" + ("" if ok or not allow_short else "; --allow-short-horizon"))]


def gate_g1(df: pd.DataFrame) -> list:
    """Relabel invariance: R1 and R3 returns == on every level (and rollout) for each member
    pair; R0, R2, R4-CDLS and R4 only warn."""
    out = []
    for a, b in RELABEL_PAIRS:
        for policy in G1_STRICT + G1_WARN:
            x, y = ([df[(df["member"] == m) & (df["policy"] == policy)]
                     .set_index(["level_id", "rollout"])["unshaped_return"] for m in (a, b)])
            if x.empty or y.empty:
                if policy in G1_STRICT and (a in set(df["member"]) or b in set(df["member"])):
                    out.append(("G1", "SKIP", f"{policy} {a} vs {b}: not run on both"))
                continue
            fail = "FAIL" if policy in G1_STRICT else "WARN"
            if not x.index.sort_values().equals(y.index.sort_values()):
                out.append(("G1", fail, f"{policy} {a} vs {b}: different levels/rollouts played"))
                continue
            y = y.loc[x.index]
            unequal = int((x.to_numpy() != y.to_numpy()).sum())
            detail = (f"{policy} {a} vs {b}: {len(x) - unequal}/{len(x)} equal, "
                      f"max |diff| {np.abs(x.to_numpy() - y.to_numpy()).max():.3g}")
            out.append(("G1", "PASS" if unequal == 0 else fail, detail))
    return out


def g2_readings(g_rand: pd.Series, g_ref: pd.Series) -> dict:
    gap = g_ref - g_rand
    sd_rand = g_rand.std(ddof=1)
    return {"mean_gap": gap.mean(), "min_gap": gap.min(), "sd_rand": sd_rand,
            "ratio": gap.mean() / sd_rand, "paired": gap.mean() / gap.std(ddof=1),
            "max_sd": max(g_ref.std(ddof=1), sd_rand)}


def gate_g2(df: pd.DataFrame) -> list:
    """Per member: mean_c(G_ref - G_rand) > 5 sd_c(G_rand) and min_c(G_ref - G_rand) > 0 fails;
    the paired and max-sd readings warn. G_ref is the R4 row; without R4-CDLS rows the rule
    family's best stands in and the reading is marked provisional (it never fails)."""
    out = []
    keys = ["member", "split", "level_id"]
    g_rand_all = df[df["policy"] == "R0"].groupby(keys)["unshaped_return"].mean()
    ref = df[df["policy"] == "R4"]
    provisional = ref.empty
    if provisional:
        ref = build_r4(df, require_cdls=False)
    g_ref_all = ref.set_index(keys)["unshaped_return"]
    for member in sorted(set(df["member"])):
        if member not in g_rand_all.index.get_level_values(0) or \
                member not in g_ref_all.index.get_level_values(0):
            out.append(("G2", "SKIP", f"{member}: needs R0 and a reference"))
            continue
        g_rand, g_ref = g_rand_all.loc[member], g_ref_all.loc[member]
        both = g_rand.index.intersection(g_ref.index)
        r = g2_readings(g_rand.loc[both], g_ref.loc[both])
        ok = r["mean_gap"] > G2_FACTOR * r["sd_rand"] and r["min_gap"] > 0
        label = "provisional (rule-family best as G_ref, no R4-CDLS)" if provisional else "G_ref = R4"
        status = "PASS" if ok else ("WARN" if provisional else "FAIL")
        out.append(("G2", status, f"{member} [{label}]: mean gap {r['mean_gap']:.4f} = "
                    f"{r['ratio']:.1f} sd(G_rand) ({r['sd_rand']:.4f}), min gap {r['min_gap']:.4f}, "
                    f"{len(both)} levels"))
        if r["paired"] <= G2_FACTOR:
            out.append(("G2", "WARN", f"{member}: paired reading mean/sd(G_ref - G_rand) = "
                        f"{r['paired']:.1f} <= {G2_FACTOR:g}"))
        if r["mean_gap"] <= G2_FACTOR * r["max_sd"]:
            out.append(("G2", "WARN", f"{member}: max-sd reading mean gap / max(sd(G_ref), "
                        f"sd(G_rand)) = {r['mean_gap'] / r['max_sd']:.1f} <= {G2_FACTOR:g}"))
    return out


def gate_g3(df: pd.DataFrame, rho: dict) -> list:
    """Headroom: on every member with rho >= 1 - 1e-6, mean_c(zc_ceiling - G_ref) >= 0.05 (G_ref
    the R4 row); the per-context minimum is reported."""
    ref = df[df["policy"] == "R4"]
    if ref.empty:
        return [("G3", "SKIP", "no R4 rows")]
    out = []
    for member in sorted(set(ref["member"])):
        r = rho.get(member)
        if r is None or r < G3_RHO:
            out.append(("G3", "SKIP", f"{member}: rho {r:.3f} < 1, not gated"))
            continue
        gap = (ref["zc_ceiling"] - ref["unshaped_return"])[ref["member"] == member]
        ok = gap.mean() >= G3_MIN_GAP
        out.append(("G3", "PASS" if ok else "FAIL",
                    f"{member} (rho {r:.3f}): mean_c(zc - G_ref) {gap.mean():.4f} (needs >= "
                    f"{G3_MIN_GAP}), min {gap.min():.4f}, {len(gap)} levels"))
    return out


def gate_g4(df: pd.DataFrame) -> list:
    out = []
    not_terminated = int((~df["terminated"].astype(bool)).sum())
    out.append(("G4", "PASS" if not_terminated == 0 else "FAIL",
                f"{len(df) - not_terminated}/{len(df)} rows terminated"))
    out.append(("G4", "PASS", "no illegal actions: ReferencePredictor raises on any action "
                "outside the mask"))
    keys = ["member", "split", "level_id"]
    r4 = df[df["policy"] == "R4"].set_index(keys)["unshaped_return"]
    if len(r4):
        family = df[df["policy"].isin(("R1", "R2") + R3_FAMILY) & (df["rollout"] == 0)]
        best = family.groupby(keys)["unshaped_return"].max().reindex(r4.index)
        played = family.groupby(keys)["policy"].nunique().reindex(r4.index, fill_value=0)
        below = int((r4 < best).sum())
        out.append(("G4", "PASS" if below == 0 else "FAIL",
                    f"R4 >= max(R1, R2, R3 family) on {len(r4) - below}/{len(r4)} contexts"))
        partial = int((played < 2 + len(R3_FAMILY)).sum())
        if partial:
            out.append(("G4", "WARN", f"{partial}/{len(r4)} R4 contexts lack part of the rule "
                        f"family (R1, R2 and the four R3 variants); R4 there guards only what "
                        f"was played"))
        cdls = df[df["policy"] == R4_POLICY]
        divergent = int((cdls["r4_divergent_steps"].fillna(0) > 0).sum())
        out.append(("G4", "PASS" if divergent == 0 else "WARN",
                    f"R4-CDLS diverged from its plan on {divergent}/{len(cdls)} levels"))
    return out


def run_gates(df: pd.DataFrame, horizon: int, allow_short: bool, rho: dict) -> bool:
    results = gate_horizon(horizon, allow_short) + gate_g1(df) + gate_g2(df) + gate_g3(df, rho) \
        + gate_g4(df)
    print("\ngate           status  detail")
    for gate, status, detail in results:
        print(f"{gate:14s} {status:6s}  {detail}")
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"{len(failed)} gate check(s) failed" if failed else "all gates pass")
    return not failed


# ─── Outputs ────────────────────────────────────────────────────────────────────────────

def paths(out_dir: str, split: str) -> dict:
    return {"rows": os.path.join(out_dir, f"references_{split}.csv"),
            "summary": os.path.join(out_dir, f"references_summary_{split}.csv"),
            "meta": os.path.join(out_dir, "references_meta.json")}


def load_rows(path: str) -> pd.DataFrame:
    # round_trip: pandas' default float parser can be an ulp off, which would break G1's ==
    # between rows read back from disk and rows just played.
    if not os.path.exists(path):
        return pd.DataFrame(columns=ROW_COLUMNS)
    return pd.read_csv(path, float_precision="round_trip")


def load_meta(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def merge_rows(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """old without the (policy, member, split) triples new re-ran and without derived R4 rows,
    plus new, plus R4 rebuilt from the result."""
    rerun = set(zip(new["policy"], new["member"], new["split"]))
    keep = np.array([(p, m, s) not in rerun and p != "R4"
                     for p, m, s in zip(old["policy"], old["member"], old["split"])], dtype=bool)
    parts = [frame for frame in (old.loc[keep], new) if len(frame)]
    df = pd.concat(parts, ignore_index=True)[ROW_COLUMNS] if parts else new[ROW_COLUMNS]
    r4 = build_r4(df)
    if len(r4):
        df = pd.concat([df, r4], ignore_index=True)
    return df.sort_values(["member", "policy", "level_id", "rollout"], kind="stable") \
        .reset_index(drop=True)


def check_meta_compatible(meta: dict, jar_sha: str, horizon: int) -> None:
    if meta.get("jar_sha256") not in (None, jar_sha):
        raise SystemExit(f"--out-dir holds results from another jar ({meta['jar_sha256']}); "
                         f"use a fresh --out-dir")
    if meta.get("H") not in (None, horizon):
        raise SystemExit(f"--out-dir holds results for H = {meta['H']}; use a fresh --out-dir")


def write_outputs(out_dir: str, split: str, new_rows: list, meta_update: dict, rho: dict) -> pd.DataFrame:
    p = paths(out_dir, split)
    new = pd.DataFrame(new_rows).reindex(columns=ROW_COLUMNS)
    df = merge_rows(load_rows(p["rows"]), new)
    df.to_csv(p["rows"], index=False)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(p["rows"][:-4] + ".parquet", index=False)
    except ImportError:
        pass
    summarise(df, rho).to_csv(p["summary"], index=False)

    meta = load_meta(p["meta"])
    params_hashes = {**meta.get("params_hash", {}), **meta_update.pop("params_hash")}
    runs = meta.get("runs", []) + [meta_update.pop("run")]
    meta.update(meta_update, params_hash=params_hashes, runs=runs)
    with open(p["meta"], "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return df


# ─── Main ───────────────────────────────────────────────────────────────────────────────

def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--members", default="all", help="all, or a comma-separated list of ids")
    ap.add_argument("--split", default="test", choices=sorted(levels.EVAL_SPLITS))
    ap.add_argument("--policies", default="R0,R1,R2,R3,R3-fam,R4",
                    help="R0,R1,R2,R3,R3-f1,R3-c0,R3-c1; R3-fam = the four R3 variants; "
                         "R4 = R4-CDLS plus the derived R4 row")
    ap.add_argument("--rollouts", type=int, default=8, help="R0 passes over the split")
    ap.add_argument("--num-cpu", type=int, default=16, help="simulator workers (JVMs)")
    ap.add_argument("--plan-workers", type=int, default=32,
                    help="processes for the R4 plans and the ceilings")
    ap.add_argument("--plan-cache", default=r4_planner.CACHE_DIR,
                    help="R4-CDLS plan cache (<dir>/<member>/<level>.json)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--max-episode-length", type=int, default=None,
                    help="H; defaults to config.yml's max_episode_length")
    ap.add_argument("--check", action="store_true", help="run the gates after playing")
    ap.add_argument("--check-only", action="store_true", help="run the gates on --out-dir")
    ap.add_argument("--strict-r4", action="store_true", help="R4 raises on any plan mismatch")
    ap.add_argument("--allow-short-horizon", action="store_true",
                    help="the H_consistency gate warns instead of failing; playing still needs "
                         f"H >= {levels.HORIZON} (utils.misc.level_stream)")
    ap.add_argument("--unlock-lockbox", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.split == "lockbox" and not args.unlock_lockbox:
        print("the lockbox split is sealed; pass --unlock-lockbox to play it", file=sys.stderr)
        return 2
    common = load_common(args.config)
    horizon = args.max_episode_length or int(common["max_episode_length"])
    p = paths(args.out_dir, args.split)

    manifest = load_manifest()
    known = [m["id"] for m in manifest["members"]]
    with open(levels.REPO_ANCHOR_PATH) as f:
        anchor = json.load(f)
    rho = {m: member_rho(manifest, m, anchor) for m in known}
    if args.check_only:
        df = load_rows(p["rows"])
        if df.empty:
            print(f"no rows in {p['rows']}", file=sys.stderr)
            return 2
        return 0 if run_gates(df, load_meta(p["meta"]).get("H", horizon),
                              args.allow_short_horizon, rho) else 1

    members = known if args.members == "all" else [m.strip() for m in args.members.split(",")]
    unknown = sorted(set(members) - set(known))
    if unknown:
        print(f"unknown members {unknown}; known: {known}", file=sys.stderr)
        return 2
    policies = expand_policies(args.policies)
    if horizon < levels.HORIZON:
        # Checked here, before any JVM starts: level_stream raises inside a worker's factory,
        # after that worker's JVM is already running.
        print(f"H = {horizon}: RING-N levels need max_episode_length >= {levels.HORIZON} "
              f"(utils.misc.level_stream; arrivals run to t = {levels.LAST_ARRIVAL})",
              file=sys.stderr)
        return 2

    jar = os.environ.setdefault("CLOUDSIM_GATEWAY_JAR", DEFAULT_JAR)
    os.environ["JAVA_LOG_DESTINATION"] = "none"
    os.environ.setdefault("JAVA_LOG_LEVEL", "WARN")
    os.environ["SAVE_EXPERIMENT"] = "false"
    jar_sha = sha256_file(jar)
    meta = load_meta(p["meta"])
    check_meta_compatible(meta, jar_sha, horizon)
    os.makedirs(args.out_dir, exist_ok=True)
    git_sha, dirty = git_state()
    code = code_sha256()

    print(f"references: members {members}, split {args.split}, policies {policies}, "
          f"R0 rollouts {args.rollouts}, {args.num_cpu} workers, H = {horizon}", flush=True)
    # The gRPC port block is picked per member, just before its JVMs start; plans ignore it.
    params_of = {member: member_params(common, member, manifest, args.split, args.num_cpu,
                                       horizon, base_port=None) for member in members}
    start = time.perf_counter()
    prepared = prepare_levels(params_of, args.split, R4_POLICY in policies, args)
    prepare_s = time.perf_counter() - start
    df = None
    for member in members:
        start = time.perf_counter()
        params = dict(params_of[member], grpc_base_port=free_port_block(args.num_cpu))
        rows, timings = run_member(member, params, policies, args, prepared[member])
        elapsed = time.perf_counter() - start
        print(f"  {member}: {elapsed:.1f} s", flush=True)
        plans = [p_["plan"] for p_ in prepared[member].values() if p_["plan"] is not None]
        df = write_outputs(args.out_dir, args.split, rows, {
            "git_sha": git_sha, "git_dirty": dirty, "jar": os.path.relpath(jar, REPO),
            "jar_sha256": jar_sha, "PORT_VERSION": r4_planner.cp.PORT_VERSION,
            "R0_SEED": R0_SEED, "guard_band_g": r4_planner.R4_GUARD_BAND,
            "r4_plan_version": r4_planner.PLAN_VERSION,
            "H": horizon, "numpy_version": np.__version__, "python_version": platform.python_version(),
            "decisions": DECISIONS, "params_hash": {member: params_hash(params)},
            "run": {"member": member, "split": args.split, "policies": policies,
                    "git_sha": git_sha, "git_dirty": dirty, "code_sha256": code,
                    "rollouts": args.rollouts, "num_cpu": args.num_cpu,
                    "elapsed_s": round(elapsed, 1), "passes": timings,
                    "levels_prepared_s": round(prepare_s, 1),
                    "r4_planning_s": [p_["stats"]["planning_s"] for p_ in plans],
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
        }, rho)
    print(f"wrote {p['rows']}, {p['summary']}, {p['meta']}", flush=True)
    if args.check:
        return 0 if run_gates(df, horizon, args.allow_short_horizon, rho) else 1
    return 0


def _raise_on_sigterm(signum, _frame):
    # As entrypoint.py: unwind, so env.close() stops the worker JVMs instead of orphaning them.
    raise KeyboardInterrupt(f"terminated by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    sys.exit(main())
