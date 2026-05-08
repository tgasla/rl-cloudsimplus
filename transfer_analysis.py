"""
Transfer analysis: B → C (50k steps)
Computes all standard transferability metrics used in top-tier RL transfer papers.

Usage:
    python3 transfer_analysis.py
"""

import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────────

BASE = "common/logs/euromlsys"

TRANSFER = {
    "euromlsys":   f"{BASE}/extractor_comparison_train_b_100k_transfer_to_c_50k/euromlsys/monitor.csv",
    "attention":   f"{BASE}/extractor_comparison_train_b_100k_transfer_to_c_50k/attention_pooling/monitor.csv",
    "turret":      f"{BASE}/extractor_comparison_train_b_100k_transfer_to_c_50k/turret/monitor.csv",
}

ORACLE = {
    "euromlsys": f"{BASE}/oracle_train_c_50k/euromlsys/monitor.csv",
    "attention":  f"{BASE}/oracle_train_c_50k/attention_pooling/monitor.csv",
    "turret":     f"{BASE}/oracle_train_c_50k/turret/monitor.csv",
}

TRAIN_B = {
    "euromlsys":   f"{BASE}/extractor_comparison_train_b_100k/euromlsys/monitor.csv",
    "attention":   f"{BASE}/extractor_comparison_train_b_100k/attention_pooling/monitor.csv",
    "turret":      f"{BASE}/extractor_comparison_train_b_100k/turret/monitor.csv",
}

THRESHOLD_FRAC = 0.80   # time-to-80%-of-oracle-peak
FINAL_WINDOW   = 50     # episodes for "final convergence"
SMOOTH_K       = 30     # rolling window for smooth curve stats

# ── Loader ───────────────────────────────────────────────────────────────────

def load_rewards(path: str) -> np.ndarray:
    rewards = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:          # skip metadata comment + header
        line = line.strip()
        if line:
            rewards.append(float(line.split(",")[0]))
    return np.array(rewards)

# ── Metrics ──────────────────────────────────────────────────────────────────

def auc(rewards: np.ndarray) -> float:
    """Trapezoidal AUC per episode (normalised by count)."""
    return float(np.trapz(rewards) / len(rewards))

def smooth(rewards: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(rewards, np.ones(k) / k, mode="valid")

def time_to_threshold(rewards: np.ndarray, threshold: float) -> int:
    """First episode index (1-based) at or above threshold. -1 if never reached."""
    idxs = np.where(rewards >= threshold)[0]
    return int(idxs[0] + 1) if len(idxs) else -1

def compute_transfer_metrics(transfer: np.ndarray, oracle: np.ndarray) -> dict:
    oracle_peak   = float(oracle.max())
    threshold     = THRESHOLD_FRAC * oracle_peak
    n             = min(len(transfer), len(oracle))
    t_trim        = transfer[:n]
    o_trim        = oracle[:n]

    norm_jump     = transfer[0] / oracle_peak
    auc_ratio     = auc(t_trim) / auc(o_trim)
    ep_80         = time_to_threshold(transfer, threshold)
    final_mean    = float(transfer[-FINAL_WINDOW:].mean())
    final_vs_peak = final_mean / oracle_peak
    jumpstart_gap = transfer[0] - oracle[0]     # raw advantage over oracle at step 0

    return {
        "jumpstart":       float(transfer[0]),
        "oracle_start":    float(oracle[0]),
        "jumpstart_gap":   jumpstart_gap,        # how much head-start vs oracle
        "norm_jumpstart":  norm_jump,
        "oracle_peak":     oracle_peak,
        "transfer_peak":   float(transfer.max()),
        "norm_peak":       float(transfer.max()) / oracle_peak,
        "final_mean":      final_mean,
        "final_vs_peak":   final_vs_peak,
        "auc_ratio":       auc_ratio,
        "ep_to_80pct":     ep_80,
        "threshold":       threshold,
        "n_episodes":      len(transfer),
    }

# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(v, pct=False, ep=False):
    if v == -1:
        return "never"
    if ep:
        return f"{v:,}"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"

SEP  = "─" * 72
SEP2 = "═" * 72

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    oracle_r   = {k: load_rewards(v) for k, v in ORACLE.items()}
    transfer_r = {k: load_rewards(v) for k, v in TRANSFER.items()}
    train_b_r  = {k: load_rewards(v) for k, v in TRAIN_B.items()}

    # Each extractor is normalised against its own oracle (self-normalisation principle)
    metrics = {k: compute_transfer_metrics(transfer_r[k], oracle_r[k])
               for k in transfer_r}

    print()
    print(SEP2)
    print("  TRANSFER ANALYSIS  —  Env B → Env C  (B=100k train, C=50k transfer)")
    print("  Self-normalisation: each extractor divided by its own oracle ceiling")
    print(SEP2)

    # ── Source training summary ──────────────────────────────────────────────
    print()
    print("SOURCE TRAINING QUALITY  (Env B, 100k steps)")
    print(SEP)
    print(f"  {'Extractor':<16} {'Peak reward':>12}  {'Final-50 mean':>14}  {'Episodes':>10}")
    print(SEP)
    for name in ["euromlsys", "attention", "turret"]:
        r = train_b_r[name]
        print(f"  {name:<16} {r.max():>12.3f}  {r[-50:].mean():>14.3f}  {len(r):>10,}")
    print()

    # ── Per-extractor oracle summary ─────────────────────────────────────────
    print("ORACLES  (Env A from scratch, 50k steps — per-extractor ceiling)")
    print(SEP)
    print(f"  {'Extractor':<16} {'Start':>8}  {'Peak':>8}  {'Final-50':>10}  {'80% thresh':>12}")
    print(SEP)
    for name in ["euromlsys", "attention", "turret"]:
        r = oracle_r[name]
        peak = float(r.max())
        print(f"  {name:<16} {r[0]:>8.3f}  {peak:>8.3f}  {r[-FINAL_WINDOW:].mean():>10.3f}  {0.8*peak:>12.3f}")
    print()

    # ── Transfer metrics table ───────────────────────────────────────────────
    print("TRANSFER METRICS  (B → A, 50k fine-tuning steps)")
    print(SEP)
    col = 14
    h = (f"  {'Metric':<26}"
         f"{'euromlsys':>{col}}"
         f"{'attention':>{col}}"
         f"{'turret':>{col}}")
    print(h)
    print("  " + "─" * (26 + 3 * col))

    rows = [
        ("Zero-shot jumpstart",         "jumpstart",      False, False),
        ("Oracle start (scratch ep-1)", "oracle_start",   False, False),
        ("Jumpstart advantage",          "jumpstart_gap",  False, False),
        ("Norm jumpstart (÷ oracle pk)", "norm_jumpstart", True,  False),
        ("Peak reward (transfer)",       "transfer_peak",  False, False),
        ("Norm peak (÷ oracle peak)",    "norm_peak",      True,  False),
        (f"Final-{FINAL_WINDOW} mean",  "final_mean",     False, False),
        ("Final vs oracle peak",        "final_vs_peak",  True,  False),
        ("AUC ratio (transfer/oracle)", "auc_ratio",      False, False),
        (f"Episodes to 80% of oracle",  "ep_to_80pct",    False, True),
    ]

    for label, key, pct, ep in rows:
        vals = [metrics[name][key] for name in ["euromlsys", "attention", "turret"]]
        print(f"  {label:<26}" + "".join(f"{fmt(v, pct=pct, ep=ep):>{col}}" for v in vals))

    print()

    # ── Ranking ──────────────────────────────────────────────────────────────
    print("RANKING SUMMARY")
    print(SEP)
    criteria = [
        ("Best zero-shot jumpstart",  "jumpstart",      False),
        ("Best norm jumpstart",        "norm_jumpstart", False),
        ("Best AUC ratio",             "auc_ratio",      False),
        ("Best final convergence",     "final_mean",     False),
        ("Fastest to 80% oracle",      "ep_to_80pct",    True),   # lower is better
    ]
    for label, key, lower_better in criteria:
        vals = {name: metrics[name][key] for name in ["euromlsys", "attention", "turret"]}
        if lower_better:
            # -1 (never) means worst
            def sort_key(item):
                v = item[1]
                return float("inf") if v == -1 else v
            winner = min(vals.items(), key=sort_key)[0]
        else:
            winner = max(vals.items(), key=lambda x: x[1])[0]
        val_str = ", ".join(f"{n}={fmt(v, ep=lower_better)}" for n, v in vals.items())
        print(f"  {label:<30}  → {winner:<16} ({val_str})")

    print()
    print(SEP2)
    print("  INTERPRETATION")
    print(SEP2)
    print()
    print("  norm_jumpstart > 1.0  → transferred knowledge overshoots oracle (impossible long-term,")
    print("                           but positive = strong zero-shot warm start)")
    print("  auc_ratio > 1.0       → transfer accumulates more reward per episode than scratch")
    print("  ep_to_80pct           → 'never' means the method never reached 80% of oracle ceiling")
    print("                           within the 50k-step budget")
    print()


if __name__ == "__main__":
    main()
