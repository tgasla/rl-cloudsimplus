"""
Transfer analysis: B→A, B→C, C→A, C→B.
Computes all standard transferability metrics used in top-tier RL transfer papers.

Usage:
    python3 transfer_analysis.py
"""

import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────────

BASE = "common/logs/euromlsys"

EXTRACTORS = [
    "euromlsys", "attention", "turret", "hybrid", "spane", "type_stratified",
    "hybrid_v2", "type_stratified_v2",
    "type_stratified_scaled", "type_stratified_pre_head_scaled",
    "hybrid_scaled", "hybrid_pre_head_scaled",
    "rbf",
    "type_stratified_embed",
    "fusion",
    "attention_idfree",
    "pma",
    "hierarchical",
    "hybrid_rbf",
    "swat",
    "aria",
    "tsar",
]

_LOG_NAMES = {
    "euromlsys":                      "euromlsys",
    "attention":                      "attention_pooling",
    "turret":                         "turret",
    "hybrid":                         "hybrid",
    "spane":                          "spane",
    "type_stratified":                "type_stratified",
    "hybrid_v2":                      "hybrid_pre_head",
    "type_stratified_v2":             "type_stratified_pre_head",
    "type_stratified_scaled":         "type_stratified_scaled",
    "type_stratified_pre_head_scaled":"type_stratified_pre_head_scaled",
    "hybrid_scaled":                  "hybrid_scaled",
    "hybrid_pre_head_scaled":         "hybrid_pre_head_scaled",
    "rbf":                            "rbf",
    "type_stratified_embed":          "type_stratified_embed",
    "fusion":                         "fusion",
    "attention_idfree":               "attention_idfree",
    "pma":                            "pma",
    "hierarchical":                   "hierarchical",
    "hybrid_rbf":                     "hybrid_rbf",
    "swat":                           "swat",
    "aria":                           "aria",
    "tsar":                           "tsar",
}

def _path(subdir, name):
    return f"{BASE}/{subdir}/{_LOG_NAMES[name]}/monitor.csv"

# ── 300k oracle paths — correct budget for paper (shared by all analysis blocks) ─
# 50k oracles proved too short: transfer peaks exceeded oracle peaks.
# Each extractor has its own oracle (apples-to-apples: transfer vs same-arch scratch).
ORACLE_A_300k = {k: _path("oracle_train_a_300k", k) for k in EXTRACTORS}
ORACLE_B_300k = {k: _path("oracle_train_b_300k", k) for k in EXTRACTORS}
ORACLE_C_300k = {k: _path("oracle_train_c_300k", k) for k in EXTRACTORS}

# ── 1M oracle paths — the CONVERGED ceiling (use for paper headline numbers) ──
# NOTE a deliberate difference from the 300k oracles above: these are ONE oracle per
# environment, trained with the euromlsys reference architecture, not one per extractor.
# So norm_* here answers "how close does transfer get to the best known from-scratch
# policy", NOT "…to the same architecture trained from scratch".
# Why 1M: the 50k and 300k oracles were still climbing (Env C 50k: peak 12.102 vs
# final-50 10.865), which let transfer curves exceed them — norm_jumpstart > 100%,
# an invalid result. Env C 1M closes that gap to 0.058.
ORACLE_A_1M = {k: _path("oracle_train_a_1m", "euromlsys") for k in EXTRACTORS}
ORACLE_C_1M = {k: _path("oracle_train_c_1m", "euromlsys") for k in EXTRACTORS}

# ── B-source legacy paths (100k training — curves not converged, for reference only) ──
TRAIN_B   = {k: _path("extractor_comparison_train_b_100k", k) for k in EXTRACTORS}
TRANSFER_A = {k: _path("extractor_comparison_train_b_100k_transfer_to_a_50k", k) for k in EXTRACTORS}
ORACLE_A   = {k: _path("oracle_train_a_50k", k) for k in EXTRACTORS}   # 50k — underconverged
TRANSFER_C = {k: _path("extractor_comparison_train_b_100k_transfer_to_c_50k", k) for k in EXTRACTORS}
ORACLE_C   = {k: _path("oracle_train_c_50k", k) for k in EXTRACTORS}   # 50k — underconverged
TRANSFER = TRANSFER_C  # legacy alias
ORACLE   = ORACLE_C    # legacy alias

# ── B-source 300k paths (converged source training — use for paper final results) ──
# All B-source curves still rising at 100k (confirmed empirically: final-50 < peak).
# Key extractors only; add others here when reruns complete.
B_SOURCE_300k_EXTRACTORS = [
    "euromlsys", "attention", "spane", "fusion", "attention_idfree", "hybrid", "type_stratified",
]
TRAIN_B_300k    = {k: _path("extractor_comparison_train_b_300k", k) for k in B_SOURCE_300k_EXTRACTORS}
TRANSFER_A_300k = {k: _path("extractor_comparison_train_b_300k_transfer_to_a_50k", k) for k in B_SOURCE_300k_EXTRACTORS}
TRANSFER_C_300k = {k: _path("extractor_comparison_train_b_300k_transfer_to_c_50k", k) for k in B_SOURCE_300k_EXTRACTORS}

# ── C-source paths (source = Env C, 500k training, 100k fine-tuning) ─────────
C_SOURCE_EXTRACTORS = ["euromlsys", "attention", "fusion", "aria", "swat", "tsar"]
TRAIN_C           = {k: _path("extractor_comparison_train_c_500k", k) for k in C_SOURCE_EXTRACTORS}
TRANSFER_A_from_C = {k: _path("extractor_comparison_train_c_500k_transfer_to_a_100k", k) for k in C_SOURCE_EXTRACTORS}
TRANSFER_B_from_C = {k: _path("extractor_comparison_train_c_500k_transfer_to_b_100k", k) for k in C_SOURCE_EXTRACTORS}

THRESHOLD_FRAC = 0.80   # time-to-80%-of-oracle-peak
FINAL_WINDOW   = 50     # episodes for "final convergence"
SMOOTH_K       = 30     # rolling window for smooth curve stats

# Zero-shot budget: every episode completing before the first gradient update, i.e.
# n_rollout_steps * num_cpu env-steps (config.yml: 128 * 16). Using transfer[0] alone
# is a single sample against a per-episode sd of ~0.47, which is why rankings computed
# from it invert under any averaging. Averaging the whole pre-update rollout costs
# nothing and cuts the SEM by ~8x while still measuring a policy that has not learned.
PRE_UPDATE_STEPS = 128 * 16

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

def load_run(path: str):
    """Return (rewards, lengths) from a monitor.csv. Columns are r,l,t."""
    rewards, lengths = [], []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        rewards.append(float(parts[0]))
        lengths.append(int(float(parts[1])))
    return np.array(rewards), np.array(lengths)


def zero_shot_stats(path: str, budget: int = PRE_UPDATE_STEPS) -> dict:
    """Mean reward over episodes completing within the first `budget` env-steps.

    These episodes are all generated by the transferred policy before its first
    gradient update, so their mean is a genuine zero-shot measurement rather than
    a one-episode sample.
    """
    rewards, lengths = load_run(path)
    cutoff = int(np.searchsorted(np.cumsum(lengths), budget, side="right"))
    cutoff = max(cutoff, 1)
    window = rewards[:cutoff]
    sem = float(window.std(ddof=1) / np.sqrt(len(window))) if len(window) > 1 else float("nan")
    return {"mean": float(window.mean()), "n": int(len(window)), "sem": sem,
            "first": float(rewards[0])}


def normalise(value: float, floor: float, ceiling: float) -> float:
    """Floor-to-ceiling normalisation. floor = untrained performance in the target env."""
    span = ceiling - floor
    return float("nan") if span <= 0 else (value - floor) / span


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
    oracle_peak   = float(smooth(oracle, SMOOTH_K).max())
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

def run_analysis(title, oracle_paths, transfer_paths, names, source_paths=None, source_label="",
                 shared_oracle=False):
    """Print a full transfer analysis block for one direction."""
    def _load(paths):
        out = {}
        for k, p in paths.items():
            if k in names:
                try:
                    out[k] = load_rewards(p)
                except FileNotFoundError:
                    pass
        return out

    oracle_r   = _load(oracle_paths)
    transfer_r = _load(transfer_paths)

    available = [n for n in names if n in transfer_r and n in oracle_r]
    if not available:
        print(f"\n  [skipped — no data for {title}]\n")
        return

    metrics = {k: compute_transfer_metrics(transfer_r[k], oracle_r[k])
               for k in available}

    print()
    print(SEP2)
    print(f"  TRANSFER ANALYSIS  —  {title}")
    print("  Normalisation: one shared oracle ceiling for every extractor"
          if shared_oracle else
          "  Self-normalisation: each extractor divided by its own oracle ceiling")
    print(SEP2)

    # ── Source training summary ──────────────────────────────────────────────
    if source_paths is not None:
        source_r = _load(source_paths)
        lbl = f"  ({source_label})" if source_label else ""
        print()
        print(f"SOURCE TRAINING QUALITY{lbl}")
        print(SEP)
        print(f"  {'Extractor':<16} {'Peak reward':>12}  {'Final-50 mean':>14}  {'Episodes':>10}")
        print(SEP)
        for name in available:
            if name in source_r:
                r = source_r[name]
                print(f"  {name:<16} {r.max():>12.3f}  {r[-50:].mean():>14.3f}  {len(r):>10,}")
        print()

    # ── Per-extractor oracle summary ─────────────────────────────────────────
    env_label = title.split("→")[-1].strip().split()[0]
    _ceiling = "shared euromlsys 1M ceiling" if shared_oracle else "per-extractor ceiling"
    print(f"ORACLES  (Env {env_label} from scratch — {_ceiling}, smoothed peak)")
    print(SEP)
    print(f"  {'Extractor':<16} {'Start':>8}  {'Peak':>8}  {'Final-50':>10}  {'80% thresh':>12}")
    print(SEP)
    for name in available:
        r = oracle_r[name]
        peak = float(smooth(r, SMOOTH_K).max())
        print(f"  {name:<16} {r[0]:>8.3f}  {peak:>8.3f}  {r[-FINAL_WINDOW:].mean():>10.3f}  {0.8*peak:>12.3f}")
    print()

    # ── Transfer metrics table ───────────────────────────────────────────────
    col = 14
    print(f"TRANSFER METRICS  ({title})")
    print(SEP)
    header = f"  {'Metric':<26}" + "".join(f"{n:>{col}}" for n in available)
    print(header)
    print("  " + "─" * (26 + col * len(available)))

    rows = [
        ("Zero-shot jumpstart",         "jumpstart",      False, False),
        ("Oracle start (scratch ep-1)", "oracle_start",   False, False),
        ("Jumpstart advantage",         "jumpstart_gap",  False, False),
        ("Norm jumpstart (÷ oracle pk)","norm_jumpstart", True,  False),
        ("Peak reward (transfer)",      "transfer_peak",  False, False),
        ("Norm peak (÷ oracle peak)",   "norm_peak",      True,  False),
        (f"Final-{FINAL_WINDOW} mean", "final_mean",     False, False),
        ("Final vs oracle peak",       "final_vs_peak",  True,  False),
        ("AUC ratio (transfer/oracle)","auc_ratio",      False, False),
        ("Episodes to 80% of oracle",  "ep_to_80pct",    False, True),
    ]

    for label, key, pct, ep in rows:
        vals = [metrics[name][key] for name in available]
        print(f"  {label:<26}" + "".join(f"{fmt(v, pct=pct, ep=ep):>{col}}" for v in vals))

    print()

    # ── Ranking ──────────────────────────────────────────────────────────────
    print("RANKING SUMMARY")
    print(SEP)
    criteria = [
        ("Best zero-shot jumpstart",  "jumpstart",      False),
        ("Best norm jumpstart",       "norm_jumpstart", False),
        ("Best AUC ratio",            "auc_ratio",      False),
        ("Best final convergence",    "final_mean",     False),
        ("Fastest to 80% oracle",     "ep_to_80pct",    True),
    ]
    for label, key, lower_better in criteria:
        vals = {name: metrics[name][key] for name in available}
        if lower_better:
            winner = min(vals.items(), key=lambda x: float("inf") if x[1] == -1 else x[1])[0]
        else:
            winner = max(vals.items(), key=lambda x: x[1])[0]
        val_str = ", ".join(f"{n}={fmt(v, ep=lower_better)}" for n, v in vals.items())
        print(f"  {label:<30}  → {winner:<16} ({val_str})")
    print()


def main():
    # ── Legacy B-source 100k (underconverged source — for reference / existing results) ──
    run_analysis("B → A (downscale, 100k source — legacy)", ORACLE_A, TRANSFER_A, EXTRACTORS,
                 source_paths=TRAIN_B, source_label="Env B, 100k source training")
    run_analysis("B → C (upscale,   100k source — legacy)", ORACLE_C, TRANSFER_C, EXTRACTORS,
                 source_paths=TRAIN_B, source_label="Env B, 100k source training")

    # ── B-source 300k (converged source — paper final results) ────────────────────────
    run_analysis("B → A (downscale, 300k source)", ORACLE_A_300k, TRANSFER_A_300k, B_SOURCE_300k_EXTRACTORS,
                 source_paths=TRAIN_B_300k, source_label="Env B, 300k source training")
    run_analysis("B → C (upscale,   300k source)", ORACLE_C_300k, TRANSFER_C_300k, B_SOURCE_300k_EXTRACTORS,
                 source_paths=TRAIN_B_300k, source_label="Env B, 300k source training")

    # ── B-source 300k vs the CONVERGED 1M oracles — paper headline numbers ───────────
    run_analysis("B → A (downscale, 300k source, 1M oracle)", ORACLE_A_1M, TRANSFER_A_300k,
                 B_SOURCE_300k_EXTRACTORS,
                 source_paths=TRAIN_B_300k, source_label="Env B, 300k source training",
                 shared_oracle=True)
    run_analysis("B → C (upscale,   300k source, 1M oracle)", ORACLE_C_1M, TRANSFER_C_300k,
                 B_SOURCE_300k_EXTRACTORS,
                 source_paths=TRAIN_B_300k, source_label="Env B, 300k source training",
                 shared_oracle=True)

    # ── C-source 500k (converged source — paper final results) ───────────────────────
    run_analysis("C → A (structural downscale, 500k source)", ORACLE_A_300k, TRANSFER_A_from_C, C_SOURCE_EXTRACTORS,
                 source_paths=TRAIN_C, source_label="Env C, 500k source training")
    run_analysis("C → B (size downscale,       500k source)", ORACLE_B_300k, TRANSFER_B_from_C, C_SOURCE_EXTRACTORS,
                 source_paths=TRAIN_C, source_label="Env C, 500k source training")

    print(SEP2)
    print("  INTERPRETATION")
    print(SEP2)
    print()
    print("  norm_jumpstart > 1.0  → transferred knowledge overshoots oracle start")
    print("  auc_ratio > 1.0       → transfer accumulates more reward per episode than scratch")
    print("  ep_to_80pct           → 'never' = never reached 80% of oracle ceiling in budget")
    print()
    print("  Legacy 100k B-source: all curves still rising — do NOT use for paper final numbers")
    print("  300k/50k oracles: still underconverged — norm_jumpstart can exceed 100% (invalid)")
    print("  1M oracles: converged ceiling — use the '1M oracle' blocks for paper headline numbers")
    print("  1M oracle is ONE per environment (euromlsys arch), shared by all extractors")
    print()


if __name__ == "__main__":
    main()
