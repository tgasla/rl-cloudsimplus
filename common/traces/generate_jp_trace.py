#!/usr/bin/env python3
"""
Realistic job-placement trace generator for rl-cloudsimplus.

Design principles:
  - Origins are micro DCs ONLY. Cloud/edge DCs can disappear on transfer
    (e.g. Env B→A removes cloud_dc_rotterdam), which would make trace
    entries for those origins invalid. Micro DCs are guaranteed present
    in all transfer environments by construction.
  - Core distribution is inspired by Huawei East-1 / VMAgent (heavy tail
    towards small jobs: ~65% ≤4 cores, ~25% 6-10, ~10% 12-16).
  - Bursty Poisson arrivals with a two-peak load profile (morning/afternoon),
    matching realistic cloud workload patterns.
  - Mixed delay-sensitivity and deadline profiles: mostly tolerant batch
    jobs, some moderate, few latency-critical SLA jobs.

Trace CSV columns:
  job_id, arrival_time, mi, required_cores, location,
  delay_sensitivity, deadline

Usage:
  # Env B trace (2 micro DC origins — ucd, dcu)
  python3 generate_jp_trace.py \\
      --n_jobs 2000 --max_time 120 --seed 42 \\
      --locations micro_dc_ucd micro_dc_dcu \\
      --output jp_b_2000.csv

  # Env C trace (3 micro DC origins — ucd, dcu, aau)
  python3 generate_jp_trace.py \\
      --n_jobs 2000 --max_time 120 --seed 42 \\
      --locations micro_dc_ucd micro_dc_dcu micro_dc_aau \\
      --output jp_c_2000.csv

  # Small fixed trace for immediate use (no Java chunk-sampling needed)
  python3 generate_jp_trace.py \\
      --n_jobs 80 --max_time 80 --seed 42 \\
      --locations micro_dc_ucd micro_dc_dcu \\
      --output jp_b_80.csv

Chunk-sampling note:
  Generate the large (2000-job) traces now; use the first-N slices for
  current fixed-episode experiments. Once Java reset() supports random
  window sampling (pending task), switch to the full 2000-job traces
  for stochastic training episodes.
"""
import argparse
import numpy as np
import pandas as pd


# ── Core distribution ─────────────────────────────────────────────────────────
# Discrete core counts and unnormalised sampling weights.
# Targets: ~65% small (1-4 cores), ~25% medium (6-10), ~10% large (12-16).
# Calibrated from Huawei East-1 cluster trace (VMAgent dataset): integer CPU
# requests with heavy small-job skew and sparse large jobs.
_CORE_CHOICES = [1, 2, 4,  4,  6,  8, 10, 12, 16]
_CORE_WEIGHTS = [25, 20, 15, 10, 10,  8,  7,  3,  2]  # unnormalised

# ── MI calibration ────────────────────────────────────────────────────────────
# CloudSim Plus: mi = required_cores * pe_mips * runtime_seconds.
# pe_mips = 60 (from topology). Target runtime: U[1, 10] simulation seconds.
# So mi = cores * 60 * U[1, 10].
_MIPS = 60

# ── Delay sensitivity ─────────────────────────────────────────────────────────
_SENSITIVITY_CHOICES = ["tolerant", "moderate", "critical"]
_SENSITIVITY_WEIGHTS = [0.60, 0.30, 0.10]

# Deadline ranges (simulation timesteps until violation). 0 = no hard deadline.
_DEADLINE_RANGES = {
    "critical": (1, 4),    # tight SLA: must complete in 1-4 timesteps
    "moderate": (3, 10),   # flexible: 3-10 timestep window
    "tolerant": (0, 15),   # best-effort: often no hard deadline (0)
}
# Fraction of tolerant jobs with no hard deadline (deadline=0)
_TOLERANT_NO_DEADLINE_PROB = 0.40


def _sample_cores(n: int, rng: np.random.Generator) -> np.ndarray:
    w = np.array(_CORE_WEIGHTS, dtype=float)
    w /= w.sum()
    return rng.choice(_CORE_CHOICES, size=n, p=w)


def _sample_mi(cores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    runtime = rng.integers(1, 11, size=len(cores))
    return (cores * _MIPS * runtime).astype(int)


def _sample_sensitivity(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(_SENSITIVITY_CHOICES, size=n, p=_SENSITIVITY_WEIGHTS)


def _sample_deadlines(sensitivities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    deadlines = np.zeros(len(sensitivities), dtype=int)
    for i, s in enumerate(sensitivities):
        lo, hi = _DEADLINE_RANGES[s]
        if s == "tolerant" and rng.random() < _TOLERANT_NO_DEADLINE_PROB:
            deadlines[i] = 0
        else:
            deadlines[i] = int(rng.integers(max(lo, 1), hi + 1))
    return deadlines


def _bursty_arrivals(n_jobs: int, max_time: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bursty Poisson arrivals with a two-peak load profile over [1, max_time].

    Lambda varies sinusoidally — simulating a morning burst and an afternoon
    burst, which matches observed cloud workload diurnal patterns. Arrivals
    are accumulated until n_jobs is reached; overflow is distributed uniformly.
    """
    t = np.linspace(0, 2 * np.pi, max_time)
    # Two peaks: primary at t≈pi/2, secondary at t≈3pi/2. Shape only.
    shape = 1.0 + 0.8 * np.sin(t) + 0.4 * np.sin(2 * t + 0.8)
    shape = np.clip(shape, 0.05, 3.0)
    # Scale so E[total arrivals] == n_jobs. Each timestep draws Poisson(λ_ts).
    # This ensures arrivals spread across the full max_time window.
    target_density = n_jobs / max_time
    lambdas = shape * (target_density / shape.mean())

    arrivals: list[int] = []
    for ts in range(1, max_time + 1):
        n = int(rng.poisson(lambdas[ts - 1]))
        arrivals.extend([ts] * n)

    # Trim excess or pad shortage (Poisson variance may land ±5%)
    while len(arrivals) < n_jobs:
        arrivals.append(int(rng.integers(1, max_time + 1)))
    arrivals = arrivals[:n_jobs]

    arrivals = np.array(arrivals[:n_jobs], dtype=int)
    arrivals.sort()
    return arrivals


def generate_trace(
    n_jobs: int,
    locations: list[str],
    max_time: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    arrival_times = _bursty_arrivals(n_jobs, max_time, rng)
    cores = _sample_cores(n_jobs, rng)
    mi = _sample_mi(cores, rng)
    sensitivities = _sample_sensitivity(n_jobs, rng)
    deadlines = _sample_deadlines(sensitivities, rng)
    locs = rng.choice(locations, size=n_jobs)

    return pd.DataFrame({
        "job_id": np.arange(n_jobs),
        "arrival_time": arrival_times,
        "mi": mi,
        "required_cores": cores,
        "location": locs,
        "delay_sensitivity": sensitivities,
        "deadline": deadlines,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a realistic job-placement trace for rl-cloudsimplus."
    )
    parser.add_argument("--n_jobs", type=int, default=2000,
                        help="Total number of jobs in the trace (default: 2000).")
    parser.add_argument("--locations", nargs="+",
                        default=["micro_dc_ucd", "micro_dc_dcu"],
                        help="Micro DC names as job origin locations. "
                             "Must match names in the target topology YAML.")
    parser.add_argument("--max_time", type=int, default=120,
                        help="Timestep window over which arrivals are spread (default: 120). "
                             "Set to ~episode_length for realistic density.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV file path.")
    args = parser.parse_args()

    df = generate_trace(args.n_jobs, args.locations, args.max_time, args.seed)
    df.to_csv(args.output, index=False)

    print(f"Generated {len(df)} jobs  →  {args.output}")
    print(f"\nCore distribution:")
    counts = df.required_cores.value_counts().sort_index()
    for cores, n in counts.items():
        bar = "█" * (n * 40 // len(df))
        print(f"  {cores:3d} cores: {n:5d}  ({100*n/len(df):5.1f}%)  {bar}")
    print(f"\nLocation distribution:")
    for loc, n in df.location.value_counts().items():
        print(f"  {loc}: {n} ({100*n/len(df):.1f}%)")
    print(f"\nSensitivity distribution:")
    for s, n in df.delay_sensitivity.value_counts().items():
        print(f"  {s}: {n} ({100*n/len(df):.1f}%)")
    print(f"\nArrival time range: {df.arrival_time.min()}–{df.arrival_time.max()}")
    print(f"Avg jobs per timestep: {len(df)/args.max_time:.1f}")


if __name__ == "__main__":
    main()
