#!/usr/bin/env python3
"""
compare_results.py — Live experiment dashboard for extractor comparison runs.

Usage:
    python3 compare_results.py [log_root]
    log_root defaults to: common/logs/euromlsys

Purpose:
    Quick overview while training is running or after experiments complete.
    Shows a training leaderboard and transfer metrics for all discovered groups.
    For deep analysis of a specific completed transfer pair, use transfer_analysis.py.

Directory layout supported:
    Two-level (current):
        <log_root>/
          <group_name>/          e.g. extractor_comparison_train_b_100k
            <extractor>/         e.g. euromlsys, attention_pooling, turret
              progress.csv
              monitor.csv

    Flat (planned — EXPERIMENTS.md):
        <log_root>/
          oracle_a_attention/    contains monitor.csv directly
          oracle_c_euromlsys/
          ...

Group auto-detection by name:
    - starts with "oracle_"          → oracle group
    - contains "_to_" or "transfer"  → transfer group
    - otherwise                      → training group

Metrics:
    Training  — rollout/ep_rew_mean from progress.csv (100-ep rolling mean, smooth)
    Transfer  — raw per-episode rewards from monitor.csv (all workers, unsmoothed)
      Jumpstart    : first episode reward (zero-shot performance before any fine-tuning)
      Norm-Jumpstart: jumpstart / oracle peak  (requires matching oracle group)
      AUC-Ratio    : AUC(transfer) / AUC(oracle) over same episode budget
      Ep-to-80%    : episodes until reward >= 80% of oracle peak (-1 = never)
"""
import sys
import os
import re
import csv

import numpy as np


# ── Readers ───────────────────────────────────────────────────────────────────

def read_progress(path: str) -> list[float]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("rollout/ep_rew_mean", "").strip()
            if v:
                try:
                    rows.append(float(v))
                except ValueError:
                    pass
    return rows


def read_monitor(path: str) -> list[float]:
    rows = []
    with open(path, newline="") as f:
        next(f)  # skip JSON comment on line 1
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(float(row["r"]))
            except (KeyError, ValueError):
                pass
    return rows


# ── Directory scanning ────────────────────────────────────────────────────────

def scan_extractors(group_path: str) -> dict[str, dict]:
    """
    Scan a group dir for per-extractor subdirs.
    Also handles flat layout where monitor.csv sits directly in group_path.
    Returns {extractor_name: {"progress": [...], "monitor": [...]}}
    """
    result = {}

    def _load(name: str, base: str):
        prog, mon = [], []
        pf = os.path.join(base, "progress.csv")
        mf = os.path.join(base, "monitor.csv")
        if os.path.isfile(pf):
            try:
                prog = read_progress(pf)
            except Exception:
                pass
        if os.path.isfile(mf):
            try:
                mon = read_monitor(mf)
            except Exception:
                pass
        if prog or mon:
            result[name] = {"progress": prog, "monitor": mon}

    # Check for flat layout (monitor.csv directly in group_path)
    if os.path.isfile(os.path.join(group_path, "monitor.csv")):
        _load("(run)", group_path)
        return result

    # Two-level layout: group_path/extractor/
    for name in sorted(os.listdir(group_path)):
        sub = os.path.join(group_path, name)
        if os.path.isdir(sub):
            _load(name, sub)
    return result


def classify_groups(root: str):
    training, transfer, oracles = [], [], []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        nl = name.lower()
        if nl.startswith("oracle"):
            oracles.append((name, path))
        elif "_to_" in nl or "transfer" in nl:
            transfer.append((name, path))
        else:
            training.append((name, path))
    return training, transfer, oracles


def extract_env(name: str) -> str | None:
    """Extract the target env letter from a group directory name."""
    # Transfer: *_to_{letter}_* or *_to_{letter}
    m = re.search(r"_to_([a-z])(?:_|$)", name, re.I)
    if m:
        return m.group(1).lower()
    # Oracle: find first single-letter segment after 'oracle' prefix
    if name.lower().startswith("oracle"):
        for part in name.split("_")[1:]:
            if len(part) == 1 and part.isalpha():
                return part.lower()
    return None


def _ext_key(name: str) -> str:
    """Normalize extractor name for oracle matching (attention_pooling → attention)."""
    return name.replace("_pooling", "")


# ── Transfer metrics ──────────────────────────────────────────────────────────

def compute_metrics(transfer: list[float], oracle: list[float]) -> dict:
    if not transfer:
        return {}
    oracle_peak = max(oracle) if oracle else None
    jumpstart = transfer[0]
    if oracle_peak and oracle_peak > 0:
        norm_jump = jumpstart / oracle_peak
        n = min(len(transfer), len(oracle))
        t_auc = float(np.trapz(transfer[:n]) / n) if n else 0.0
        o_auc = float(np.trapz(oracle[:n]) / n) if n else 0.0
        auc_ratio = t_auc / o_auc if o_auc else None
        threshold = 0.8 * oracle_peak
        ep_to = next((i for i, r in enumerate(transfer) if r >= threshold), -1)
    else:
        norm_jump = auc_ratio = ep_to = None
    return {"jumpstart": jumpstart, "norm_jump": norm_jump,
            "auc_ratio": auc_ratio, "ep_to": ep_to}


def _f(v, fmt=".4f"):
    if v is None:
        return "—"
    if v == -1:
        return "never"
    return f"{v:{fmt}}"


def _ep(v):
    if v is None:
        return "—"
    if v == -1:
        return "never"
    return str(int(v) + 1)  # convert 0-based index to 1-based episode number


SEP = "─" * 88


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "common/logs/euromlsys"
    if not os.path.isdir(root):
        print(f"Log root not found: {root}")
        return

    training, transfer, oracles = classify_groups(root)
    all_groups = training + transfer + oracles
    if not all_groups:
        print(f"No experiment groups found in {root}")
        return

    # ── Training leaderboard ──────────────────────────────────────────────────
    print(f"\n  LOG ROOT: {root}")
    print(f"\n  TRAINING LEADERBOARD  (rollout/ep_rew_mean — 100-ep rolling mean)")
    print(f"  {SEP}")
    for gname, gpath in all_groups:
        data = scan_extractors(gpath)
        if not data:
            continue
        print(f"\n  [{gname}]")
        print(f"    {'Extractor':<24} {'Rollouts':>9} {'Best':>10} {'Final':>10}")
        print(f"    {'─'*24} {'─'*9} {'─'*10} {'─'*10}")
        for ext, d in data.items():
            prog = d["progress"]
            if prog:
                print(f"    {ext:<24} {len(prog):>9} {max(prog):>10.4f} {prog[-1]:>10.4f}")
            else:
                print(f"    {ext:<24} {'—':>9} {'—':>10} {'—':>10}")

    if not transfer:
        return

    # ── Build oracle index: env → {normalized_extractor → monitor rewards} ────
    oracle_index: dict[str, dict[str, list[float]]] = {}
    for oname, opath in oracles:
        env = extract_env(oname)
        if env is None:
            continue
        oracle_index.setdefault(env, {})
        for ext, d in scan_extractors(opath).items():
            oracle_index[env][_ext_key(ext)] = d["monitor"]

    # ── Transfer metrics ──────────────────────────────────────────────────────
    print(f"\n\n  TRANSFER METRICS  (monitor.csv — raw per-episode rewards)")
    print(f"  {SEP}")

    for gname, gpath in transfer:
        tgt = extract_env(gname)
        data = scan_extractors(gpath)
        if not data:
            continue
        print(f"\n  [{gname}]  →  Env {(tgt or '?').upper()}")
        print(f"    {'Extractor':<24} {'Jump':>8} {'NormJump':>10} {'AUC-Ratio':>10} {'Ep-to-80%':>10}")
        print(f"    {'─'*24} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

        for ext, d in data.items():
            env_oracles = oracle_index.get(tgt or "", {})
            # exact match, then normalized, then any oracle for this env
            oracle_mon = (env_oracles.get(ext)
                          or env_oracles.get(_ext_key(ext))
                          or (next(iter(env_oracles.values())) if env_oracles else []))
            m = compute_metrics(d["monitor"], oracle_mon)
            if not m:
                print(f"    {ext:<24} (no monitor.csv yet)")
                continue
            norm_s = _f(m["norm_jump"], ".3f") if oracle_mon else "—(no oracle)"
            auc_s  = _f(m["auc_ratio"], ".3f") if oracle_mon else "—"
            ep_s   = _ep(m["ep_to"])           if oracle_mon else "—"
            print(f"    {ext:<24} {_f(m['jumpstart']):>8} {norm_s:>10} {auc_s:>10} {ep_s:>10}")

    if not oracle_index:
        print(f"\n  No oracle groups found in {root}.")
        print(f"  Add oracle_{{env}}_{{extractor}}/ directories to enable normalized metrics.")
    else:
        envs = ", ".join(f"Env {e.upper()}" for e in sorted(oracle_index))
        print(f"\n  Oracles available for: {envs}")

    print(f"\n  For full analysis of a specific pair, run: python3 transfer_analysis.py\n")


if __name__ == "__main__":
    main()
