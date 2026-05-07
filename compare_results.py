#!/usr/bin/env python3
"""
Compare extractor training/transfer results.

Usage:
  python3 compare_results.py [logs_root]

Defaults to common/logs/euromlsys/extractor_comparison/

Reward sources used:
  Training leaderboard  → progress.csv  rollout/ep_rew_mean  (100-ep rolling mean, smooth)
  Transfer metrics      → monitor.csv   r column             (raw per-episode, all workers)

Transfer metrics (printed when oracle experiments are present):
  Jumpstart     : first episode reward in the transfer env (monitor.csv row 0)
  Norm-Jumpstart: jumpstart / oracle_peak_reward
  AUC-Ratio     : AUC(transfer monitor curve) / AUC(oracle monitor curve), same episode budget
  Ep-to-80%     : episodes until reward >= 80% of oracle peak (-1 = never reached)
"""

import sys
import os
import csv

import numpy as np


# ── Readers ───────────────────────────────────────────────────────────────────

def read_progress(path: str) -> list[float]:
    """Return rollout/ep_rew_mean values (non-empty rows only)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("rollout/ep_rew_mean", "").strip()
            if val:
                try:
                    rows.append(float(val))
                except ValueError:
                    pass
    return rows


def read_monitor(path: str) -> list[float]:
    """Return per-episode rewards from monitor.csv (all workers, raw, unsmoothed).

    monitor.csv has a JSON comment on line 1 that must be skipped before DictReader.
    """
    rows = []
    with open(path, newline="") as f:
        next(f)  # skip "#{"t_start": ...}" comment line
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(float(row["r"]))
            except (KeyError, ValueError):
                pass
    return rows


# ── Transfer metrics ──────────────────────────────────────────────────────────

def compute_auc(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    return float(np.trapz(rewards) / len(rewards))


def compute_metrics(transfer_ep: list[float], oracle_ep: list[float]) -> dict:
    """Compute all transfer metrics from raw per-episode monitor rewards."""
    oracle_peak = max(oracle_ep) if oracle_ep else None

    jumpstart = transfer_ep[0] if transfer_ep else None

    if oracle_peak and oracle_peak > 0:
        norm_jump = jumpstart / oracle_peak if jumpstart is not None else None

        n = min(len(transfer_ep), len(oracle_ep))
        oracle_auc = compute_auc(oracle_ep[:n])
        auc_ratio = compute_auc(transfer_ep[:n]) / oracle_auc if oracle_auc else None

        threshold = 0.8 * oracle_peak
        ep_to_thresh = next(
            (i for i, r in enumerate(transfer_ep) if r >= threshold), -1
        )
    else:
        norm_jump = auc_ratio = None
        ep_to_thresh = None

    return {
        "jumpstart": jumpstart,
        "norm_jump": norm_jump,
        "auc_ratio": auc_ratio,
        "ep_to_thresh": ep_to_thresh,
    }


# ── Experiment name parsing ───────────────────────────────────────────────────

def parse_transfer_key(name: str):
    """
    Detect transfer experiments by name pattern: {extractor}_b_to_{env}.
    Returns (extractor, src_env, tgt_env) or None.
    """
    known = ["euromlsys", "turret", "attention_pooling", "attention"]
    for ext in known:
        if name.startswith(ext + "_"):
            rest = name[len(ext) + 1:]
            parts = rest.split("_to_")
            if len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) == 1:
                return ext, parts[0], parts[1]
    return None


def find_oracle_monitor(base: str, env: str, extractor: str) -> list[float]:
    """Find oracle_{env}_{extractor}/monitor.csv and return episode rewards."""
    for name in (f"oracle_{env}_{extractor}", f"oracle_{env}"):
        p = os.path.join(base, name, "monitor.csv")
        if os.path.isfile(p):
            return read_monitor(p)
    return []


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "—"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "common/logs/euromlsys/extractor_comparison"
    if not os.path.isdir(root):
        print(f"No logs yet at: {root}")
        return

    experiments = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if not experiments:
        print(f"No experiment directories found in {root}")
        return

    # ── Training leaderboard (uses progress.csv rollout means — smooth curves) ─
    print(f"\n{'Experiment':<45} {'Rollouts':>8} {'Best (smooth)':>14} {'Final (smooth)':>15}")
    print("-" * 86)
    for exp in experiments:
        csv_path = os.path.join(root, exp, "progress.csv")
        if not os.path.exists(csv_path):
            print(f"  {exp:<43} (no progress.csv yet)")
            continue
        rows = read_progress(csv_path)
        if not rows:
            print(f"  {exp:<43} {'0':>8} {'—':>14} {'—':>15}")
            continue
        print(
            f"  {exp:<43} {len(rows):>8}"
            f" {max(rows):>14.4f} {rows[-1]:>15.4f}"
        )

    # ── Transfer metrics (uses monitor.csv — raw, unsmoothed) ─────────────────
    transfer_exps = [
        (exp, parse_transfer_key(exp))
        for exp in experiments
        if parse_transfer_key(exp) is not None
    ]

    if transfer_exps:
        print(f"\n{'Transfer Metrics  (source: monitor.csv — raw episode rewards)'}")
        print(
            f"  {'Experiment':<43} {'Jump':>7} {'NormJump':>9}"
            f" {'AUC-Ratio':>10} {'Ep-to-80%':>10}"
        )
        print("  " + "-" * 83)

        no_oracle = True
        for exp, (extractor, src, tgt) in transfer_exps:
            mon_path = os.path.join(root, exp, "monitor.csv")
            if not os.path.isfile(mon_path):
                print(f"  {exp:<43} (no monitor.csv yet)")
                continue

            t_ep = read_monitor(mon_path)
            if not t_ep:
                print(f"  {exp:<43} (empty monitor.csv)")
                continue

            oracle_ep = find_oracle_monitor(root, tgt, extractor)
            m = compute_metrics(t_ep, oracle_ep)

            jump_s = _fmt(m["jumpstart"])
            norm_s = _fmt(m["norm_jump"], ".3f") if m["norm_jump"] is not None else "—(no oracle)"
            auc_s = _fmt(m["auc_ratio"], ".3f") if m["auc_ratio"] is not None else "—"
            thresh_s = str(m["ep_to_thresh"]) if m["ep_to_thresh"] is not None else "—"
            if m["ep_to_thresh"] == -1:
                thresh_s = "never"

            if oracle_ep:
                no_oracle = False

            print(
                f"  {exp:<43} {jump_s:>7} {norm_s:>9}"
                f" {auc_s:>10} {thresh_s:>10}"
            )

        if no_oracle:
            print(
                "\n  Note: oracle_{env}_{extractor} runs needed for NormJump / AUC-Ratio / Ep-to-80%."
            )

    # ── Other experiment groups ────────────────────────────────────────────────
    extra_root = "common/logs/euromlsys"
    if os.path.isdir(extra_root):
        others = [
            (d, os.path.join(extra_root, d))
            for d in sorted(os.listdir(extra_root))
            if os.path.isdir(os.path.join(extra_root, d))
            and d != os.path.basename(root)
        ]
        if others:
            print(f"\n── Other groups in {extra_root} ──────────────────────────────────")
            for group, gpath in others:
                for exp in sorted(os.listdir(gpath)):
                    p = os.path.join(gpath, exp, "progress.csv")
                    if os.path.isfile(p):
                        rows = read_progress(p)
                        best = f"{max(rows):.4f}" if rows else "—"
                        print(f"  {group}/{exp:<40} {len(rows):>8}  best={best}")


if __name__ == "__main__":
    main()
