#!/usr/bin/env python3
"""Status of a RING-N training run, against the reference ladder on the same val split.

    python3 benchmark/trainwatch.py [run_dir]      # default: the a5_S_3M headroom run

The val sweeps are the number that matters: each one plays the 24 val levels with the
deterministic policy, which is exactly how the rules in benchmark/results_r4 were scored.
Training reward (rollout/ep_rew_mean) is NOT comparable -- it is logged while the policy
still explores and on training levels.
"""
import datetime as _dt
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "common/logs/ringn/headroom/a5_S_3M")
# The ladder to compare against, read from a reference run rather than hardcoded, so this
# tracks whatever benchmark/run_references.py last produced for the member.
LADDER_DIR = os.path.join(ROOT, "benchmark/results_r4")
RULES = ("R1", "R2", "R3", "R3-f1", "R3-c0", "R3-c1")   # the full deployable family


def _ladder(member="S"):
    """(random, best deployable rule, zero-contention bound) for `member`, or None.

    The best rule is taken per level over ALL six deployable rules from the per-level file --
    the summary carries only R1/R2/R3, which understates it, since the R3 variants often win.
    """
    try:
        per = pd.read_csv(os.path.join(LADDER_DIR, "references_val.csv"))
        summ = pd.read_csv(os.path.join(LADDER_DIR, "references_summary_val.csv"))
        per, summ = per[per.member == member], summ[summ.member == member]
        if per.empty or summ.empty:
            return None
        k = summ.set_index("level_id").G_R1 / per[per.policy == "R1"].set_index("level_id").unshaped_return
        best = (per[per.policy.isin(RULES)].groupby("level_id").unshaped_return.max() * k).mean()
        return float(summ.G_rand.mean()), float(best), float(summ.zc_ceiling.mean())
    except Exception:
        return None


def main(d=DEFAULT):
    if not os.path.isdir(d):
        sys.exit(f"no such run: {d}")
    prog = pd.read_csv(os.path.join(d, "progress.csv"))
    steps = int(prog["time/total_timesteps"].iloc[-1])
    target = 3_000_000
    el = time.time() - os.path.getctime(os.path.join(d, "config.yml"))
    rate = steps / el
    left = (target - steps) / rate if rate > 0 else float("nan")

    print(f"{os.path.basename(d)}: {steps:,} / {target:,} steps "
          f"({100*steps/target:.1f}%)  {rate:.0f} steps/s  ~{left/3600:.1f} h left")

    vf = os.path.join(d, "val.csv")
    if not os.path.exists(vf):
        print("  no val sweep yet"); return
    v = pd.read_csv(vf).groupby("timestep").unshaped_return.agg(["mean", "std", "count"])
    best = v["mean"].max()
    lad = _ladder()
    if lad is None:
        print("\n  (no reference ladder yet: run benchmark/run_references.py --split val)")
        RAND, RULE, BOUND = 0.0, 1.0, 1.0
    else:
        RAND, RULE, BOUND = lad
    # wall clock of each sweep: interpolate progress.csv's time_elapsed at the sweep's timestep,
    # anchored on the present (elapsed is measured from model.learn(), not from container start)
    tp = prog[["time/total_timesteps", "time/time_elapsed"]].dropna().sort_values(
        "time/total_timesteps")                      # SB3 writes partial rows; drop them
    t0 = time.time() - float(tp["time/time_elapsed"].iloc[-1])
    el_at = np.interp(v.index.to_numpy(float),
                      tp["time/total_timesteps"].to_numpy(float),
                      tp["time/time_elapsed"].to_numpy(float))

    print(f"\n  {'sweep':>9}  {'val return':>10}  {'at':>8}  {'elapsed':>8}   vs rule   bar")
    for (t, r), e in zip(v.iterrows(), el_at):
        frac = (r["mean"] - RAND) / (RULE - RAND)
        bar = "#" * max(0, min(32, int(32 * frac)))
        ok = np.isfinite(e)
        clock = _dt.datetime.fromtimestamp(t0 + e).strftime("%H:%M:%S") if ok else "    -   "
        elap = f"{int(e)//3600:>2}h{int(e)%3600//60:02d}m" if ok else "   -  "
        print(f"  {t:>9,}  {r['mean']:>10.3f}  {clock:>8}  {elap}"
              f"   {100*frac:5.1f}%   {bar}")
    print(f"\n  best so far {best:.3f}   |  random {RAND:.3f}  best rule {RULE:.3f}  "
          f"zc bound {BOUND:.3f}")
    print(f"  the agent has closed {100*(best-RAND)/(RULE-RAND):.0f}% of the random->rule gap")
    if len(v) >= 4:
        peak_at = v["mean"].idxmax()
        stale = v.index[-1] - peak_at
        last3, prev3 = v["mean"].iloc[-3:].mean(), v["mean"].iloc[-6:-3].mean()
        # a 2-sweep lookback cannot see a peak further back; compare 3-sweep blocks instead
        print(f"  peak {best:.3f} at {peak_at:,}; {stale:,} steps since without beating it")
        print(f"  last 3 sweeps mean {last3:.3f} vs previous 3 {prev3:.3f} ({last3-prev3:+.3f})"
              f"{'  CLIMBING' if last3 - prev3 > 0.01 else '  PLATEAU' if abs(last3-prev3) <= 0.01 else '  DECLINING'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
