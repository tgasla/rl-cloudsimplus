"""RING-N workload levels. A level is a pure function of (member, level_id); nothing is pooled.

A level is generated on the fly at every episode reset, so generation is numpy-only and fast.
The rules follow the workload section of docs/analysis/05-benchmark-redesign-ring-n.json; the
core-count and arrival-intensity distributions come from the Azure anchor that
benchmark/build_anchor.py writes to common/traces/anchor_stats.json. The sensitivity mix is a
design parameter: Azure's VM category (62% "Unknown", 0.7% "Interactive") does not say how
delay-tolerant a job is, and would leave the critical class almost empty.

One timestep is 1 s, so a job's runtime on a tier is mi / tier_mips timesteps.

Validate a manifest (exits 1 if any member fails a check):
    python3 common/rl-manager/utils/levels.py --validate --manifest common/topologies/ring/manifest.json
"""
import argparse
import functools
import itertools
import json
import math
import os
import time

import numpy as np
import yaml

FIRST_ARRIVAL, LAST_ARRIVAL = 1, 160   # arrival window; the episode (H = 200) drains after it
HORIZON = 200                           # timesteps; one timestep is 1 s
SHAPE_BINS = 288                        # 5-minute bins per day in the anchor intensity shape

MIPS_REF = 60                           # edge tier: runtime_ref is measured here
RUNTIME_MEDIAN_1CORE = 11.0             # runtime_ref ~ LogNormal(ln(11 * cores^0.30), 0.85) ...
RUNTIME_CORE_EXPONENT = 0.30
RUNTIME_SIGMA = 0.85
RUNTIME_MIN, RUNTIME_MAX = 3, 60        # ... rounded, then clipped to [3, 60]

SENSITIVITIES = ("critical", "moderate", "tolerant")   # emitted by name, mapped to ints later
SENSITIVITY_MIX = {"critical": 0.20, "moderate": 0.30, "tolerant": 0.50}
SLACK = {"critical": (0, 2), "moderate": (2, 8), "tolerant": (6, 24)}   # inclusive, timesteps
# Micro is the slowest tier and has zero network delay, so a deadline of at least ceil(mi / 40)
# is met at every legal DC at zero contention: a violation is always the policy's doing.
DEADLINE_FLOOR_MIPS = 40
SITE_ZIPF_EXPONENT = 0.9

TRAIN_LEVELS = range(0, 100000)
VAL_LEVELS = range(1000000, 1000024)
TEST_LEVELS = range(2000000, 2000048)
LOCKBOX_LEVELS = range(3000000, 3000048)
EVAL_SPLITS = {"val": VAL_LEVELS, "test": TEST_LEVELS, "lockbox": LOCKBOX_LEVELS}

# --validate only: network delay is simulator config, not in the topology YAML.
NET_DELAY = {"cloud": 3.0, "edge": 1.0, "micro": 0.0}
LEGAL_DEGREE = 4                        # own DC, two ring neighbours, cloud
VALIDATE_LEVELS = 1000
W_BAR_TOLERANCE = 0.05
OFFERED_LOAD_TOLERANCE = 0.01
HEAVY_TAIL_MIN_TOP_RATIO = 1.5          # hottest site's share over the uniform share 1/n_sites
PHASE_QUARTERS = 4                      # arrival window split for the site-phase check
PHASE_CHI2_MIN = 1.25                   # level-mean chi2/df; 1.0 when every site shares one phase

HERE = os.path.dirname(os.path.abspath(__file__))
# common/traces on the host; /mgr/traces in the container, where common/docker-compose.yml mounts
# utils/ and traces/ side by side under /mgr.
_ANCHOR_CANDIDATES = (os.path.join(HERE, "..", "..", "traces", "anchor_stats.json"),
                      os.path.join(HERE, "..", "traces", "anchor_stats.json"))
REPO_ANCHOR_PATH = next((p for p in _ANCHOR_CANDIDATES if os.path.exists(p)), _ANCHOR_CANDIDATES[0])


# ─── Distributions ──────────────────────────────────────────────────────────

def _core_distribution(anchor: dict) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(anchor["cores"]["probs"], dtype=float)
    return np.asarray(anchor["cores"]["values"], dtype=np.int64), probs / probs.sum()


def _sensitivity_probs() -> np.ndarray:
    return np.array([SENSITIVITY_MIX[s] for s in SENSITIVITIES], dtype=float)


def _runtime_mu(cores):
    return np.log(RUNTIME_MEDIAN_1CORE * np.asarray(cores, dtype=float) ** RUNTIME_CORE_EXPONENT)


def expected_runtime_ref(cores: int) -> float:
    """E[clip(round(LogNormal), RUNTIME_MIN, RUNTIME_MAX)] for a job of `cores`, in closed form."""
    mu = float(_runtime_mu(cores))
    values = np.arange(RUNTIME_MIN, RUNTIME_MAX + 1)
    # round(X) clips to value k iff X < k + 0.5 (and X >= k - 0.5 unless k is the minimum)
    upper = np.append(values[:-1] + 0.5, math.inf)
    cdf = [0.5 * math.erfc(-(math.log(u) - mu) / (RUNTIME_SIGMA * math.sqrt(2))) for u in upper]
    return float(values @ np.diff(cdf, prepend=0.0))


def w_bar(anchor: dict) -> float:
    """Mean job size E[cores * runtime_ref] in core-timesteps: the load normaliser."""
    values, probs = _core_distribution(anchor)
    return float(sum(p * c * expected_runtime_ref(c) for c, p in zip(values.tolist(), probs)))


def resolve_lambda(manifest: dict, member_id: str, anchor: dict) -> float:
    """Arrival rate (jobs/timestep): {"rho": r} -> r * capacity_pes / w_bar; {"lambda_of": X} -> X's."""
    member = {m["id"]: m for m in manifest["members"]}[member_id]
    load = member["load"]
    if "lambda_of" in load:
        return resolve_lambda(manifest, load["lambda_of"], anchor)
    return load["rho"] * member["capacity_pes"] / w_bar(anchor)


# ─── Level generation ───────────────────────────────────────────────────────

def _sample_sites(n_sites: int, arrivals: np.ndarray, offset: int, shape: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Origin index per job: Zipf site popularity x per-site diurnal phase; the hot site is
    permuted per level."""
    popularity = (1.0 + rng.permutation(n_sites)) ** -SITE_ZIPF_EXPONENT
    phase = rng.integers(SHAPE_BINS, size=n_sites)
    steps = np.arange(FIRST_ARRIVAL, LAST_ARRIVAL + 1)
    weight = popularity * shape[(steps[:, None] + offset + phase) % SHAPE_BINS]
    cum = np.cumsum(weight, axis=1)[arrivals - FIRST_ARRIVAL]
    u = rng.random(len(arrivals)) * cum[:, -1]
    return np.minimum((cum <= u[:, None]).sum(axis=1), n_sites - 1)


def generate_level(member: dict, level_id: int, anchor: dict, lam: float,
                   base_seed: int = 0) -> list[dict]:
    """The jobs of one level as trace_utils descriptors, in arrival order.

    Only `location` depends on the member (through its origins); every other field depends on
    (level_id, lam, anchor, base_seed) alone, so members sharing a lambda share the job stream.
    """
    arrival_rng, job_rng, site_rng = (
        np.random.default_rng(s)
        for s in np.random.SeedSequence(entropy=base_seed, spawn_key=(level_id,)).spawn(3)
    )
    shape = np.asarray(anchor["intensity"]["shape"], dtype=float)

    # Non-homogeneous Poisson: lambda(t) = lam * s(t + offset); a uniform offset per level makes
    # the offered load lam on average over levels, since s has mean 1 over the day.
    offset = int(arrival_rng.integers(SHAPE_BINS))
    steps = np.arange(FIRST_ARRIVAL, LAST_ARRIVAL + 1)
    arrivals = np.repeat(steps, arrival_rng.poisson(lam * shape[(steps + offset) % SHAPE_BINS]))
    n = len(arrivals)

    core_values, core_probs = _core_distribution(anchor)
    cores = job_rng.choice(core_values, size=n, p=core_probs)
    runtime_ref = np.clip(
        np.rint(job_rng.lognormal(_runtime_mu(cores), RUNTIME_SIGMA)), RUNTIME_MIN, RUNTIME_MAX
    ).astype(np.int64)
    mi = MIPS_REF * runtime_ref                                   # per-PE MI
    sensitivity = job_rng.choice(len(SENSITIVITIES), size=n, p=_sensitivity_probs())
    slack_lo, slack_hi = np.array([SLACK[s] for s in SENSITIVITIES]).T
    slack = job_rng.integers(slack_lo[sensitivity], slack_hi[sensitivity], endpoint=True)
    deadline = -(-mi // DEADLINE_FLOOR_MIPS) + slack

    origins = member["origins"]
    site = _sample_sites(len(origins), arrivals, offset, shape, site_rng)

    return [
        {
            "jobId": job_id,
            "submissionDelay": t,
            "mi": m,
            "cores": c,
            "location": origins[o],
            "delaySensitivity": SENSITIVITIES[s],
            "deadline": d,
        }
        for job_id, (t, m, c, o, s, d) in enumerate(zip(
            arrivals.tolist(), mi.tolist(), cores.tolist(), site.tolist(),
            sensitivity.tolist(), deadline.tolist(),
        ))
    ]


def sample_train_level(rng: np.random.Generator) -> int:
    return int(rng.integers(TRAIN_LEVELS.start, TRAIN_LEVELS.stop))


def eval_levels(split: str, rank: int, num_workers: int) -> list[int]:
    """The ids worker `rank` of `num_workers` enumerates for an eval split."""
    return list(EVAL_SPLITS[split][rank::num_workers])


# ─── Per-episode instances ──────────────────────────────────────────────────

# The simulator's encoding of delay sensitivity (Java: 0 tolerant, 1 moderate, 2 critical).
SENSITIVITY_LEVELS = {"tolerant": 0, "moderate": 1, "critical": 2}


class LevelSource:
    """The jobs of one member's levels, in the simulator's encoding, as a jobs_json payload.

    Locations become indices into `datacenter_names`, the topology order the environment was
    built with, and sensitivities become levels. Eval splits revisit the same few levels, so
    the last `cache_size` payloads are kept.
    """

    def __init__(self, manifest_path: str, member_id: str, datacenter_names: list[str],
                 anchor_path: str = REPO_ANCHOR_PATH, cache_size: int = 64):
        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(anchor_path) as f:
            self._anchor = json.load(f)
        self._member = {m["id"]: m for m in manifest["members"]}[member_id]
        # The environment must be built on this member's own topology, DC for DC: a wrong
        # !include would otherwise train another arm of the family without any error.
        topologies_dir = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
        expected = [dc["name"] for dc in load_topology(os.path.join(topologies_dir, self._member["yaml"]))]
        if list(datacenter_names) != expected:
            raise ValueError(f"datacenters are not member {member_id}'s topology "
                             f"({self._member['yaml']}): got {list(datacenter_names)}")
        self._lam = resolve_lambda(manifest, member_id, self._anchor)
        self._index = {name: i for i, name in enumerate(datacenter_names)}
        self.jobs_json = functools.lru_cache(maxsize=cache_size)(self._jobs_json)

    def _jobs_json(self, level_id: int) -> str:
        jobs = generate_level(self._member, level_id, self._anchor, self._lam)
        for job in jobs:
            job["location"] = self._index[job["location"]]
            job["delaySensitivity"] = SENSITIVITY_LEVELS[job["delaySensitivity"]]
        return json.dumps(jobs, separators=(",", ":"))


class LevelSampler:
    """Level ids for one worker: a random train level from the worker's own stream, or the
    worker's share of an eval split in a fixed round-robin order."""

    def __init__(self, split: str, rank: int, num_workers: int, seed: int):
        if split == "train":
            rng = np.random.default_rng([seed, rank])
            self._next = lambda: sample_train_level(rng)
        else:
            ids = eval_levels(split, rank, num_workers)
            if not ids:
                raise ValueError(f"worker {rank} of {num_workers} gets no {split} level")
            cycle = itertools.cycle(ids)
            self._next = lambda: next(cycle)

    def next(self) -> int:
        return self._next()


# ─── Validation ─────────────────────────────────────────────────────────────

class _TopologyLoader(yaml.SafeLoader):
    pass


for _tag in ("!datacenter", "!host", "!vm"):
    _TopologyLoader.add_constructor(_tag, lambda loader, node: loader.construct_mapping(node, deep=True))


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def load_topology(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.load(f, Loader=_TopologyLoader)


def kendall_tau_b(x, y) -> float:
    """Kendall tau-b, exact with ties, from the contingency table of (x, y). Linear in len(x)
    plus quadratic in the number of distinct values, which is small for integer job fields."""
    xi = np.unique(x, return_inverse=True)[1]
    yi = np.unique(y, return_inverse=True)[1]
    table = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(table, (xi, yi), 1)
    n_x, n_y = table.shape
    at_least = np.zeros((n_x + 1, n_y + 1))           # [i, j] = #pairs with x-index >= i, y-index >= j
    at_least[:n_x, :n_y] = table[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
    below = np.zeros((n_x + 1, n_y + 1))              # [i, j] = #pairs with x-index >= i, y-index < j
    below[:n_x, 1:] = table[::-1].cumsum(0)[::-1].cumsum(1)
    concordant = (table * at_least[1:, 1:]).sum()
    discordant = (table * below[1:, :n_y]).sum()
    n0 = len(xi) * (len(xi) - 1) / 2
    ties_x = (table.sum(1) * (table.sum(1) - 1) / 2).sum()
    ties_y = (table.sum(0) * (table.sum(0) - 1) / 2).sum()
    return float((concordant - discordant) / math.sqrt((n0 - ties_x) * (n0 - ties_y)))


def site_time_chi2_per_df(location: np.ndarray, arrivals: np.ndarray) -> float:
    """Pearson chi2/df of one level's (origin, arrival-quarter) table: about 1 when the origin
    mix is stationary over the arrival window, above 1 when sites peak at different times."""
    site = np.unique(location, return_inverse=True)[1]
    quarter = (arrivals - FIRST_ARRIVAL) * PHASE_QUARTERS // (LAST_ARRIVAL - FIRST_ARRIVAL + 1)
    table = np.zeros((site.max() + 1, PHASE_QUARTERS))
    np.add.at(table, (site, quarter), 1)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(chi2 / ((table.shape[0] - 1) * (PHASE_QUARTERS - 1)))


def _legal_destinations(topology: list[dict]) -> dict[str, list[dict]]:
    by_name = {dc["name"]: dc for dc in topology}
    return {
        dc["name"]: [dc] + [by_name[name] for name in _as_list(dc["connect_to"])]
        for dc in topology if dc.get("connect_to")
    }


def _feasible(dc: dict, cores: np.ndarray, mi: np.ndarray, deadline: np.ndarray) -> np.ndarray:
    """Whether each job completes by its deadline on some VM of `dc` when placed at arrival."""
    ok = np.zeros(len(cores), dtype=bool)
    for host in _as_list(dc["hosts"]):
        for vm in _as_list(host["vms"]):
            completion = NET_DELAY[dc["type"]] + mi / vm["pe_mips"]
            ok |= (cores <= vm["pes"]) & (completion <= deadline)
    return ok


def validate_member(member: dict, topology: list[dict], anchor: dict, lam: float,
                    level_ids) -> dict:
    """Generate `level_ids` for `member` and measure every generator guarantee.
    Returns the measurements and a list of the checks that failed."""
    legal = _legal_destinations(topology)
    origins = member["origins"]
    n_sites = len(origins)
    fields = {k: [] for k in ("cores", "runtime_ref", "sensitivity", "deadline")}
    gen_ms, n_jobs, sorted_shares, phase_chi2, infeasible = [], [], [], [], 0
    for level_id in level_ids:
        start = time.perf_counter()
        jobs = generate_level(member, level_id, anchor, lam)
        gen_ms.append(1000 * (time.perf_counter() - start))
        n_jobs.append(len(jobs))
        cores = np.array([j["cores"] for j in jobs])
        mi = np.array([j["mi"] for j in jobs])
        deadline = np.array([j["deadline"] for j in jobs])
        location = np.array([j["location"] for j in jobs])
        fields["cores"].append(cores)
        fields["runtime_ref"].append(mi // MIPS_REF)
        fields["sensitivity"].append(np.array([j["delaySensitivity"] for j in jobs]))
        fields["deadline"].append(deadline)
        shares = np.array([(location == o).sum() for o in origins]) / max(len(jobs), 1)
        sorted_shares.append(np.sort(shares)[::-1])
        phase_chi2.append(site_time_chi2_per_df(location, np.array([j["submissionDelay"] for j in jobs])))
        for origin in origins:
            at = location == origin
            for dc in legal[origin]:
                infeasible += int((~_feasible(dc, cores[at], mi[at], deadline[at])).sum())
    cols = {k: np.concatenate(v) for k, v in fields.items()}

    target_w_bar = w_bar(anchor)
    mean_cd = float((cols["cores"] * cols["runtime_ref"]).mean())
    # Reported, not gated. A deadline is ceil(1.5 * runtime) plus a class slack, so within a
    # class it tracks runtime (tau ~0.8-0.98), as completion SLOs that scale with job size do.
    # No slack rule keeps critical deadlines tight and tau <= 0.55; whether EDF and SJF still
    # decide differently is checked on the live simulator instead.
    tau = {
        s: kendall_tau_b(cols["deadline"][cols["sensitivity"] == s],
                         cols["runtime_ref"][cols["sensitivity"] == s])
        for s in SENSITIVITIES
    }
    arrival_steps = LAST_ARRIVAL - FIRST_ARRIVAL + 1
    load_err = float(np.mean(n_jobs) / (arrival_steps * lam) - 1)
    mean_shares = np.mean(sorted_shares, axis=0)
    degrees = sorted({len(legal[o]) for o in origins})
    report = {
        "member": member["id"], "lambda": lam, "levels": len(n_jobs),
        "jobs_per_level": float(np.mean(n_jobs)),
        "gen_ms_mean": float(np.mean(gen_ms)), "gen_ms_max": float(np.max(gen_ms)),
        "w_bar": target_w_bar, "mean_cores_x_runtime": mean_cd,
        "w_bar_rel_err": mean_cd / target_w_bar - 1,
        "kendall_tau": tau,
        "legal_degrees": degrees, "infeasible_job_dc_pairs": infeasible,
        "offered_load_rel_err": load_err,
        "site_shares_sorted": mean_shares.round(4).tolist(),
        "top_site_ratio": float(mean_shares[0] * n_sites),
        "site_time_chi2_per_df": float(np.mean(phase_chi2)),
    }
    failures = []
    if abs(report["w_bar_rel_err"]) > W_BAR_TOLERANCE:
        failures.append(f"E[c*d] {mean_cd:.2f} off w_bar {target_w_bar:.2f} by more than {W_BAR_TOLERANCE:.0%}")
    if degrees != [LEGAL_DEGREE]:
        failures.append(f"legal destination counts {degrees}, expected {LEGAL_DEGREE}")
    if infeasible:
        failures.append(f"{infeasible} (job, legal DC) pairs miss the deadline at zero contention")
    if abs(load_err) > OFFERED_LOAD_TOLERANCE:
        failures.append(f"offered load off lambda by {load_err:+.2%}")
    if report["top_site_ratio"] < HEAVY_TAIL_MIN_TOP_RATIO:
        failures.append(f"hottest site carries {report['top_site_ratio']:.2f}x the uniform share")
    if report["site_time_chi2_per_df"] < PHASE_CHI2_MIN:
        failures.append(f"site peaks not phase-separated: site x arrival-quarter chi2/df "
                        f"{report['site_time_chi2_per_df']:.2f} < {PHASE_CHI2_MIN}")
    report["failures"] = failures
    return report


def validate_manifest(manifest_path: str, anchor: dict, n_levels: int) -> list[dict]:
    """Validate every member. A member's `yaml` is relative to the topologies directory, the
    parent of the manifest's directory."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    topologies_dir = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    level_ids = TRAIN_LEVELS[:n_levels]
    return [
        validate_member(member, load_topology(os.path.join(topologies_dir, member["yaml"])),
                        anchor, resolve_lambda(manifest, member["id"], anchor), level_ids)
        for member in manifest["members"]
    ]


def _print_report(r: dict) -> None:
    tau = "  ".join(f"{s}={t:.3f}" for s, t in r["kendall_tau"].items())
    print(f"[{'FAIL' if r['failures'] else 'ok'}] {r['member']}: lambda={r['lambda']:.3f} "
          f"jobs/level={r['jobs_per_level']:.0f} gen={r['gen_ms_mean']:.1f}ms (max {r['gen_ms_max']:.1f})")
    print(f"    E[c*d]={r['mean_cores_x_runtime']:.3f} vs w_bar={r['w_bar']:.3f} ({r['w_bar_rel_err']:+.2%})"
          f"   offered load {r['offered_load_rel_err']:+.2%}")
    print(f"    kendall tau(deadline, runtime): {tau}")
    print(f"    legal degree {r['legal_degrees']}, infeasible (job, DC) pairs {r['infeasible_job_dc_pairs']}")
    print(f"    site shares (level-mean, sorted) {r['site_shares_sorted']}  top/uniform {r['top_site_ratio']:.2f}"
          f"  site x time chi2/df {r['site_time_chi2_per_df']:.2f}")
    for failure in r["failures"]:
        print(f"    FAILED: {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RING-N level generation for a manifest.")
    parser.add_argument("--validate", action="store_true", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--anchor", default=REPO_ANCHOR_PATH)
    parser.add_argument("--levels", type=int, default=VALIDATE_LEVELS)
    args = parser.parse_args()
    with open(args.anchor) as f:
        anchor = json.load(f)
    reports = validate_manifest(args.manifest, anchor, args.levels)
    for r in reports:
        _print_report(r)
    failed = [r["member"] for r in reports if r["failures"]]
    print(f"{len(reports) - len(failed)}/{len(reports)} members pass" + (f"; failed: {failed}" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
