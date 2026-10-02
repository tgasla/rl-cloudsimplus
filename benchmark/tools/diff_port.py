"""Differential test: benchmark/cloudsim_port.py against the live job-placement gateway.

Plays the same action sequence through the port and through a JobPlacementEnv on its own
gateway JVM, and compares, at every step:
  - the visible slots' fingerprints (cores, nominal_runtime_ref, time_to_due, sensitivity
    one-hot, reach row) and the action mask;
  - per host free_pes (exact) and backlog_core_ts (reported exact, allowed within 1), except
    when the jar lists no hosts on the last step of an episode in which every job finished:
    its CloudSim has shut down (see P9 in cloudsim_port.py; the report's java_shut_down);
  - the step's ledger (value, penalty, cost, met, violated, expired, placed, waiting) and
    unshaped reward, and whether the episode terminated;
and at the end the unshaped return and, from the gateway's DEBUG log, every job's VM, start
and finish time.

Level mode plays RING-N levels under one or more action policies, each seeded by (seed, level):
  random  per real slot, a uniform draw over the no-op and the mask's legal DCs
  defer   the no-op with probability 0.85, else a uniform legal DC: fills the 32-slot window,
          binds jobs long after they arrive, lets them expire unplaced
  origin  the job's own DC: long queues on the small DCs
  cloud   the cloud: the longest network delay, many jobs in flight at once

    python3 benchmark/tools/diff_port.py                      # 4 S + 2 GAM-lo test levels, random
    python3 benchmark/tools/diff_port.py --policies random,defer,origin,cloud
    python3 benchmark/tools/diff_port.py --scenarios          # the unit tests' scenarios
    # write the golden fixtures:
    python3 benchmark/tools/diff_port.py --members S:2 --levels 2000000,2000001 --capture-fixtures benchmark/tests/data

Scenario mode (--scenarios) plays every hand-made scenario of benchmark/tests/test_cloudsim_port.py
(SCENARIOS, on its small PARAMS topology) and also requires the jar's start and finish times to
be exactly the ones the unit tests assert.

Exits 1 if anything differs beyond the tolerances (free_pes, fingerprints, ledger and times
exact; backlog within 1; return within 1e-9).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
RL_MANAGER = os.path.join(REPO, "common", "rl-manager")
for path in (BENCH, RL_MANAGER, os.path.join(RL_MANAGER, "gym_cloudsimplus")):
    if path not in sys.path:
        sys.path.insert(0, path)

import cloudsim_port as cp  # noqa: E402

CONFIG = os.path.join(REPO, "domain", "job-placement", "config.yml")
RING = os.path.join(REPO, "common", "topologies", "ring")
MANIFEST = os.path.join(RING, "manifest.json")
LEDGER_KEYS = ("jobs_waiting", "jobs_placed", "sla_value_realized", "sla_penalty_paid",
               "resource_cost", "jobs_met", "jobs_violated", "jobs_expired_unplaced",
               "offered_value", "unshaped_reward")
FINISH_LINE = re.compile(
    r"Cloudlet (\d+), (\d+) mi, (\d+) cores on vm(\d+)/host(\d+)/dc(\d+)\. "
    r"Arrived (\S+), started (\S+), finished (\S+), exec")
POLICIES = ("random", "defer", "origin", "cloud")
DEFER_NOOP_PROBABILITY = 0.85


# ─── Params ──────────────────────────────────────────────────────────────────

class _ConfigLoader(yaml.SafeLoader):
    """config.yml with its custom tags left unresolved: only `common:` is read."""


for _tag in ("!include", "!datacenter", "!host", "!vm"):
    _ConfigLoader.add_constructor(_tag, lambda loader, node: None)


def common_params(config: str = CONFIG) -> dict:
    with open(config) as f:
        params = dict(yaml.load(f, Loader=_ConfigLoader)["common"])
    params.update(cloudlet_to_dc_mapping="rl", rl_problem="job_placement", num_cpu=1,
                  log_dir=None, save_experiment=False)
    params.setdefault("mode", "evaluate")
    params.setdefault("num_experiments", 1)
    return params


def member_params(member: str, split: str = "test", config: str = CONFIG,
                  horizon: int | None = None) -> dict:
    """config.yml `common:` on the member's topology, as the reference runner builds it."""
    from utils import levels
    from utils.misc import (_check_datacenter_amounts_are_one, _check_datacenters_unique,
                            _translate_connect_to_names_to_idx)

    params = common_params(config)
    manifest = json.load(open(MANIFEST))
    entry = {m["id"]: m for m in manifest["members"]}[member]
    topology = levels.load_topology(os.path.join(os.path.dirname(RING), entry["yaml"]))
    for dc in topology:
        dc["connect_to"] = levels._as_list(dc.get("connect_to", []))
        dc["hosts"] = levels._as_list(dc["hosts"])
        for host in dc["hosts"]:
            host["vms"] = levels._as_list(host["vms"])
    _check_datacenters_unique(topology)
    _check_datacenter_amounts_are_one(topology)
    params.update(datacenters=_translate_connect_to_names_to_idx(topology), benchmark_member=member,
                  ring_manifest=MANIFEST, level_split=split)
    if horizon is not None:
        params["max_episode_length"] = horizon
    return params


def level_jobs_json(params: dict, level: int) -> str:
    from utils import levels
    source = levels.LevelSource(params["ring_manifest"], params["benchmark_member"],
                                [dc["name"] for dc in params["datacenters"]])
    return source.jobs_json(level)


def _unit_tests():
    """benchmark/tests/test_cloudsim_port.py, for its SCENARIOS and PARAMS."""
    tests = os.path.join(BENCH, "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import test_cloudsim_port
    return test_cloudsim_port


def scenario_params(config: str = CONFIG) -> dict:
    """config.yml `common:` with the unit tests' small topology and settings on top."""
    params = common_params(config)
    params.update(json.loads(json.dumps(_unit_tests().PARAMS)))
    return params


# ─── Gateway ─────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Gateway:
    """A gateway JVM on a free port; with a log path it logs at DEBUG (every finish) there.

    Each JVM gets a log.simDir of its own (log_dir, removed on close): a gateway that logs writes
    its logback config there and parses it back at start-up, and JVMs sharing the default
    <working directory>/logs could parse it while another had just truncated it, and exit."""

    def __init__(self, jar: str = cp.JAR_PATH, log_path: str | None = None):
        self.port = _free_port()
        level, dest = ("DEBUG", "stdout") if log_path else ("WARN", "none")
        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.log_dir = tempfile.mkdtemp(prefix="diff_port_gateway_")
        self.proc = subprocess.Popen(
            ["java", f"-Dlog.level={level}", f"-Dlog.destination={dest}",
             "-Dlog.saveExperiment=false", f"-Dlog.simDir={self.log_dir}",
             "-jar", jar, "--grpc", str(self.port)],
            stdout=self._log, stderr=subprocess.STDOUT,
            env={**os.environ, "JAVA_TOOL_OPTIONS": "-XX:+UseSerialGC -Xmx512m"})
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.close()
                raise RuntimeError(f"gateway exited with {self.proc.returncode}")
            with socket.socket() as s:
                if s.connect_ex(("localhost", self.port)) == 0:
                    return
            time.sleep(0.1)
        self.close()
        raise RuntimeError("gateway did not start")

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=20)
        if self._log is not subprocess.DEVNULL:
            self._log.close()
        shutil.rmtree(self.log_dir, ignore_errors=True)


def java_cloudlets(log_path: str) -> dict:
    """{job id: (global vm id, start, finish)} from the gateway's DEBUG finish lines."""
    out = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = FINISH_LINE.search(line)
            if m:
                out[int(m.group(1))] = (int(m.group(4)), float(m.group(8)), float(m.group(9)))
    return out


# ─── Action policies ─────────────────────────────────────────────────────────

def make_policy(name: str, rng: np.random.Generator):
    """choose(ep, visible job indices, [K, D] action mask) -> actions[K]."""

    def legal_dcs(mask, i):
        return np.flatnonzero(mask[i, 1:]) + 1

    def choose(ep, vis, mask):
        action = np.zeros(mask.shape[0], dtype=np.int64)
        for i, j in enumerate(vis):
            if name == "random":
                legal = np.flatnonzero(mask[i])
                action[i] = legal[rng.integers(len(legal))]
                continue
            legal = legal_dcs(mask, i)
            if not len(legal):
                continue                      # no DC can hold the job: only the no-op
            if name == "defer":
                if rng.random() >= DEFER_NOOP_PROBABILITY:
                    action[i] = legal[rng.integers(len(legal))]
                continue
            want = ([ep.jobs[j].loc + 1] if name == "origin" else
                    [k for k in legal if ep.specs[k - 1].type == "cloud"])
            want = [k for k in want if mask[i, k]]
            action[i] = want[0] if want else legal[rng.integers(len(legal))]
        return action

    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {POLICIES}")
    return choose


def schedule_policy(binds: dict):
    """choose() that plays binds {step: [(jobId, action)]} by the port's slot of each job."""

    def choose(ep, vis, mask):
        action = np.zeros(mask.shape[0], dtype=np.int64)
        slot = {ep.jobs[j].id: i for i, j in enumerate(vis)}
        for jid, k in binds.get(ep.step_index, []):
            action[slot[jid]] = k
        return action

    return choose


# ─── One episode ─────────────────────────────────────────────────────────────

def _reach_mask_rows(ep: cp.Episode, visible: list[int], K: int, D: int):
    """The env's reach_mask and action mask, recomputed from the port's view of the slots."""
    reach = np.zeros((K, D), dtype=bool)
    mask = np.zeros((K, D), dtype=bool)
    reach[:, 0] = mask[:, 0] = True
    n = len(ep.specs)
    for i, j in enumerate(visible):
        job = ep.jobs[j]
        origin = ep.specs[job.loc]
        dests = range(n) if not origin.connect_to else [job.loc] + origin.connect_to
        for d in dests:
            if d + 1 < D:
                reach[i, d + 1] = True
                mask[i, d + 1] = ep.specs[d].max_vm_pes >= job.cores
    return reach, mask


def diff_episode(params: dict, jobs_json: str, choose, jar: str = cp.JAR_PATH,
                 log_path: str | None = None) -> dict:
    """Play one episode through the port and a live gateway and compare them (see the module
    docstring). choose(ep, visible, mask) picks the actions from the env's mask, which must equal
    the port's. Returns the report; its "problems" list is empty when they agree."""
    from gym_cloudsimplus.envs.job_placement import JobPlacementEnv

    tmp = None
    if log_path is None:
        tmp = tempfile.mkdtemp(prefix="diff_port_")
        log_path = os.path.join(tmp, "gateway.log")
    gateway = Gateway(jar, log_path)
    env = None
    problems = []
    report = {"problems": problems}
    try:
        env = JobPlacementEnv(params, jobs_as_json="[]", port=gateway.port)
        K, D = env.max_jobs_waiting, env.max_datacenters
        ep = cp.Episode(params, jobs_json)
        obs, _ = env.reset(options={"jobs_json": jobs_json})
        ret_java = ret_port = 0.0
        backlog_max = backlog_off = steps = 0
        binds = []
        per_step = {"unshaped_reward": [], "free_pes_by_dc": []}
        t0 = time.time()
        done = False
        while not done:
            s = ep.step_index
            vis = ep.visible_idx()
            # the slots: fingerprints and masks
            jobs_obs = obs["jobs_waiting_state"].reshape(K, 6)
            want = np.zeros((K, 6), dtype=np.int64)
            wire = ep.jobs_obs()
            for i in range(len(vis)):
                row = wire[i * cp.JOB_WIRE_FEATURES:(i + 1) * cp.JOB_WIRE_FEATURES]
                want[i] = [row[0]] + row[2:]
            reach, mask = _reach_mask_rows(ep, vis, K, D)
            env_mask = np.array(env.action_masks()).reshape(K, D)
            if not (jobs_obs == want).all():
                problems.append(f"step {s}: slot fingerprints differ")
            if not (obs["reach_mask"].reshape(K, D).astype(bool) == reach).all():
                problems.append(f"step {s}: reach rows differ")
            if not (env_mask == mask).all():
                problems.append(f"step {s}: action masks differ")
            action = choose(ep, vis, env_mask)
            if not env_mask[np.arange(K), action].all():
                raise ValueError(f"step {s}: the policy chose an illegal action")
            binds.extend([s, ep.jobs[vis[i]].id, int(action[i])] for i in range(len(vis)) if action[i])
            obs, _, terminated, truncated, info = env.step(action)
            mine = ep.step(action)
            done = terminated or truncated
            steps += 1
            ret_java += info["unshaped_reward"]
            ret_port += mine["unshaped_reward"]
            for key in LEDGER_KEYS:
                if info[key] != mine[key]:
                    problems.append(f"step {s}: {key} java {info[key]!r} port {mine[key]!r}")
            if bool(terminated) != mine["terminated"] or truncated:
                problems.append(f"step {s}: terminated java {terminated}/{truncated} port {mine['terminated']}")
            hosts = obs["infrastructure_state"].reshape(-1, cp.HOST_OBS_FEATURES)
            rows = np.array(ep.infra_obs(), dtype=np.int64).reshape(-1, cp.HOST_OBS_FEATURES)
            # Once every job has finished, CloudSim can run out of events inside the last step and
            # shut down; the jar then lists no hosts, which the port does not model (P9)
            shut_down = bool(terminated and not hosts.any()
                             and all(ep.finish_time(job.idx) is not None for job in ep.jobs))
            if not shut_down:
                real = hosts[: len(rows)]
                if (hosts[len(rows):] != 0).any() or not (real[:, :4] == rows[:, :4]).all():
                    problems.append(f"step {s}: host rows differ (ids, capacity or free_pes)")
                delta = np.abs(real[:, 4] - rows[:, 4])
                backlog_max = max(backlog_max, int(delta.max()))
                backlog_off += int((delta > 0).sum())
            per_step["unshaped_reward"].append(info["unshaped_reward"])
            per_step["free_pes_by_dc"].append(
                [int(sum(r[3] for r in dc.host_rows(ep.clock))) for dc in ep.dcs])
        report.update(steps=steps, return_java=ret_java, return_port=ret_port,
                      return_residual=abs(ret_java - ret_port), backlog_max_residual=backlog_max,
                      backlog_rows_off=backlog_off, java_shut_down=shut_down,
                      seconds=time.time() - t0)
        if backlog_max > 1:
            problems.append(f"backlog off by {backlog_max}")
        if abs(ret_java - ret_port) > 1e-9:
            problems.append(f"return java {ret_java!r} port {ret_port!r}")
    finally:
        if env is not None:
            env.close()
        gateway.close()
    # every job's VM, start and finish
    java = java_cloudlets(log_path)
    vm_base = np.cumsum([0] + [len(sp.vm_pes) for sp in ep.specs])
    port = {}
    for job in ep.jobs:
        b = ep.bound.get(job.idx)
        if b is not None:
            dc = ep.dcs[b[0]]
            port[job.id] = (int(vm_base[b[0]] + b[1]), dc.start.get(job.idx), dc.finish.get(job.idx))
    finished_port = {k: v for k, v in port.items() if v[2] is not None}
    fin_res = [abs(finished_port[k][2] - java[k][2]) for k in finished_port if k in java]
    start_res = [abs(finished_port[k][1] - java[k][1]) for k in finished_port if k in java]
    report.update(
        jobs=len(ep.jobs), placed=len(port), finished_java=len(java), finished_port=len(finished_port),
        expired_unplaced=sum(1 for job in ep.jobs if job.idx not in ep.bound),
        finish_mismatches=sum(r != 0 for r in fin_res), finish_max_residual=max(fin_res, default=0.0),
        start_mismatches=sum(r != 0 for r in start_res), start_max_residual=max(start_res, default=0.0),
        vm_mismatches=sum(finished_port[k][0] != java[k][0] for k in finished_port if k in java),
    )
    if set(java) != set(finished_port):
        problems.append(f"finished sets differ: java-only {sorted(set(java) - set(finished_port))[:5]} "
                        f"port-only {sorted(set(finished_port) - set(java))[:5]}")
    if report["finish_mismatches"] or report["start_mismatches"] or report["vm_mismatches"]:
        problems.append(f"{report['finish_mismatches']} finishes, {report['start_mismatches']} starts, "
                        f"{report['vm_mismatches']} VMs differ")
    report["_java"], report["_binds"], report["_per_step"] = java, binds, per_step
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return report


def play_level(member: str, level: int, seed: int, policy: str = "random", split: str = "test",
               jar: str = cp.JAR_PATH, horizon: int | None = None, fixture_dir: str | None = None,
               keep_log: str | None = None) -> dict:
    params = member_params(member, split, horizon=horizon)
    jobs_json = level_jobs_json(params, level)
    choose = make_policy(policy, np.random.default_rng([seed, level]))
    report = diff_episode(params, jobs_json, choose, jar, keep_log)
    report.update(member=member, level=level, seed=seed, policy=policy)
    if fixture_dir:
        write_fixture(fixture_dir, params, member, level, seed, jobs_json, report["_binds"],
                      report["_java"], report["_per_step"], report["return_java"])
    return report


def play_scenario(name: str, jar: str = cp.JAR_PATH) -> dict:
    jobs, binds, times = _unit_tests().SCENARIOS[name]
    report = diff_episode(scenario_params(), json.dumps(jobs), schedule_policy(binds), jar)
    java_times = {jid: (start, finish) for jid, (_, start, finish) in report["_java"].items()}
    if java_times != times:
        report["problems"].append(f"the jar's times {java_times} are not the unit test's {times}")
    report.update(member="scenario", level=name, seed=0, policy="scheduled")
    return report


def write_fixture(out_dir: str, params: dict, member: str, level: int, seed: int, jobs_json: str,
                  binds: list, java: dict, per_step: dict, ret_java: float) -> str:
    """Golden trace of one level: the binds, and what the live jar made of them."""
    jobs = json.loads(jobs_json)
    keep = ("timestep_interval", "min_time_between_events", "max_episode_length", "max_jobs_waiting",
            "mips_ref", "cloudlet_to_vm_mapping", "split_large_jobs", "reward_shaping")
    keep += tuple(k for k in params if k.startswith(("sla_value_", "sla_penalty_", "cost_", "network_delay_")))
    fixture = {
        "captured_with": {"jar_sha256": cp.jar_sha256(), "port_version": cp.PORT_VERSION,
                          "tool": "benchmark/tools/diff_port.py", "policy": "random legal",
                          "seed": seed},
        "member": member, "level_id": level,
        "params": {**{k: params[k] for k in sorted(keep)}, "datacenters": params["datacenters"]},
        "job_fields": ["jobId", "submissionDelay", "mi", "cores", "location", "delaySensitivity", "deadline"],
        "jobs": [[j[k] for k in ("jobId", "submissionDelay", "mi", "cores", "location",
                                 "delaySensitivity", "deadline")] for j in jobs],
        "binds": binds,                                          # [step, jobId, action]
        "java_cloudlets": sorted([k, *v] for k, v in java.items()),   # [jobId, vm, start, finish]
        "java_unshaped_reward": per_step["unshaped_reward"],
        "java_free_pes_by_dc": per_step["free_pes_by_dc"],
        "java_return": ret_java,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"golden_{member}_{level}.json")
    with open(path, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _summary(reports: list) -> dict:
    def worst(key, rows):
        return max((r[key] for r in rows), default=0.0)

    out = {
        "max_return_residual": worst("return_residual", reports),
        "max_finish_residual": worst("finish_max_residual", reports),
        "max_start_residual": worst("start_max_residual", reports),
        "max_backlog_residual": worst("backlog_max_residual", reports),
        "finish_mismatches": sum(r["finish_mismatches"] for r in reports),
        "vm_mismatches": sum(r["vm_mismatches"] for r in reports),
        "episodes": len(reports),
        "episodes_with_problems": sum(bool(r["problems"]) for r in reports),
        "jobs_finished": sum(r["finished_java"] for r in reports),
        "jobs_expired_unplaced": sum(r["expired_unplaced"] for r in reports),
        "by_policy": {},
    }
    for policy in sorted({r["policy"] for r in reports}):
        rows = [r for r in reports if r["policy"] == policy]
        out["by_policy"][policy] = {
            "episodes": len(rows), "max_return_residual": worst("return_residual", rows),
            "max_finish_residual": worst("finish_max_residual", rows),
            "max_backlog_residual": worst("backlog_max_residual", rows),
            "episodes_with_problems": sum(bool(r["problems"]) for r in rows),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", default="S:4,GAM-lo:2",
                    help="member:count pairs; each plays the first `count` levels of the split")
    ap.add_argument("--levels", default=None, help="explicit level ids (comma separated) for every member")
    ap.add_argument("--split", default="test")
    ap.add_argument("--policies", default="random", help=f"comma separated, from {','.join(POLICIES)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--jar", default=cp.JAR_PATH)
    ap.add_argument("--workers", type=int, default=6, help="episodes played in parallel (one JVM each)")
    ap.add_argument("--scenarios", action="store_true",
                    help="play the unit tests' hand-made scenarios instead of levels")
    ap.add_argument("--out", default=os.path.join(BENCH, "results", "diff_port.json"))
    ap.add_argument("--capture-fixtures", default=None, metavar="DIR")
    args = ap.parse_args()

    from utils import levels
    sha = cp.jar_sha256(args.jar)
    if sha != cp.JAR_SHA256:
        print(f"WARNING: jar sha256 {sha} is not the one the port was validated on ({cp.JAR_SHA256})")
    workers = max(1, args.workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        if args.scenarios:
            futures = [pool.submit(play_scenario, name, args.jar) for name in sorted(_unit_tests().SCENARIOS)]
        else:
            tasks = []
            for pair in args.members.split(","):
                member, _, count = pair.partition(":")
                ids = ([int(x) for x in args.levels.split(",")] if args.levels
                       else list(levels.EVAL_SPLITS[args.split])[: int(count or 1)])
                tasks += [(member, level, policy) for policy in args.policies.split(",") for level in ids]
            futures = [pool.submit(play_level, m, lv, args.seed, pol, args.split, args.jar, args.horizon,
                                   args.capture_fixtures) for m, lv, pol in tasks]
        reports = [f.result() for f in futures]
    for r in reports:
        for key in ("_java", "_binds", "_per_step"):
            r.pop(key, None)
    print(f"{'member':8} {'level':>14} {'policy':>9} {'steps':>5} {'jobs':>5} {'unplaced':>8} "
          f"{'return java':>20} {'|d return|':>10} {'|d finish|':>10} {'finish!=':>8} {'backlog':>7} problems")
    for r in reports:
        print(f"{r['member']:8} {r['level']!s:>14} {r['policy']:>9} {r['steps']:>5} {r['jobs']:>5} "
              f"{r['expired_unplaced']:>8} {r['return_java']:>20.15f} {r['return_residual']:>10.2e} "
              f"{r['finish_max_residual']:>10.2e} {r['finish_mismatches']:>8} {r['backlog_max_residual']:>7} "
              f"{len(r['problems'])}")
        for p in r["problems"][:10]:
            print("    ", p)
    summary = {"jar_sha256": sha, "port_version": cp.PORT_VERSION, "split": args.split,
               "seed": args.seed, "mode": "scenarios" if args.scenarios else "levels",
               "policies": {"random": "per real slot, uniform over the no-op and the mask's legal DCs",
                            "defer": f"no-op with probability {DEFER_NOOP_PROBABILITY}, else a uniform legal DC",
                            "origin": "the job's own DC", "cloud": "the cloud DC",
                            "scheduled": "the unit test's binds"},
               **_summary(reports), "episodes_detail": reports}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{summary['episodes']} episodes, {summary['episodes_with_problems']} with problems; "
          f"max |d return| {summary['max_return_residual']:.3e}, max |d finish| "
          f"{summary['max_finish_residual']:.3e} s, max |d start| {summary['max_start_residual']:.3e} s, "
          f"max |d backlog| {summary['max_backlog_residual']}; written to {args.out}")
    sys.exit(1 if summary["episodes_with_problems"] else 0)


if __name__ == "__main__":
    main()
