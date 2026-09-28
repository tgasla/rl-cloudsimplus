#!/usr/bin/env python3
"""Calibrate the RING-N workload anchor from Azure Public Dataset v1 (Cortez et al., SOSP'17).

Downloads the vmtable into benchmark/data/ (gitignored) unless it is already there, and writes
common/traces/anchor_stats.json, which common/rl-manager/utils/levels.py reads:

  cores        vm virtual core count, capped at 8, as a distribution over {1, 2, 4, 8}
  sensitivity  vm category: Interactive -> critical, Unknown -> moderate, Delay-insensitive -> tolerant
  intensity    VM creations per 5-minute bin of the day, averaged over the trace's days, mean 1

Rows are the VMs created during the trace. VMs stamped as created at t=0 were already running
when the trace began (their creation time is censored), so they are dropped.

    python3 benchmark/build_anchor.py
"""
import hashlib
import json
import math
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "common", "rl-manager"))

from utils import levels  # noqa: E402

# Listed in https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1Links.txt
VMTABLE_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1/"
               "trace_data_vmtable_vmtable.csv.gz")
DATA_PATH = os.path.join(HERE, "data", "vmtable.csv.gz")
OUT_PATH = os.path.join(REPO, "common", "traces", "anchor_stats.json")

# Field order from the dataset's schema.csv (vmtable/vmtable.csv.gz, fields 1-11); no header row.
VMTABLE_COLUMNS = ["vm_id", "subscription_id", "deployment_id", "created", "deleted",
                   "max_cpu", "avg_cpu", "p95_max_cpu", "category", "cores", "memory_gb"]
BIN_SECONDS = 300
CORE_VALUES = [1, 2, 4, 8]
# "Unkown" is the trace's own spelling.
CATEGORY_TO_SENSITIVITY = {"Interactive": "critical", "Unkown": "moderate",
                           "Delay-insensitive": "tolerant"}


def _download() -> None:
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print(f"downloading {VMTABLE_URL}")
    urllib.request.urlretrieve(VMTABLE_URL, DATA_PATH + ".part")
    os.replace(DATA_PATH + ".part", DATA_PATH)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probs(counts: pd.Series, order: list) -> list[float]:
    return [float(counts.get(k, 0) / counts.sum()) for k in order]


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    observed = pd.crosstab(a, b).to_numpy(dtype=float)
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / observed.sum()
    chi2 = ((observed - expected) ** 2 / expected).sum()
    return float(math.sqrt(chi2 / (observed.sum() * (min(observed.shape) - 1))))


def _intensity_shape(created: pd.Series) -> list[float]:
    slot = (created // BIN_SECONDS).to_numpy()
    assert (slot.max() + 1) % levels.SHAPE_BINS == 0, "trace does not cover whole days"
    counts = np.bincount(slot, minlength=slot.max() + 1).reshape(-1, levels.SHAPE_BINS)
    days_observed = np.full(levels.SHAPE_BINS, counts.shape[0])
    days_observed[0] -= 1                  # the first slot of day 0 holds only the dropped t=0 rows
    shape = counts.sum(0) / days_observed
    return [round(v, 6) for v in (shape / shape.mean()).tolist()]


def main() -> None:
    _download()
    df = pd.read_csv(DATA_PATH, header=None, names=VMTABLE_COLUMNS,
                     usecols=["created", "category", "cores"])
    censored = df.created < BIN_SECONDS
    used = df[~censored]
    assert set(used.category) <= set(CATEGORY_TO_SENSITIVITY), set(used.category)
    assert set(used.cores) <= set(CORE_VALUES) | {16}, set(used.cores)

    capped_cores = used.cores.clip(upper=max(CORE_VALUES))
    sensitivity = used.category.map(CATEGORY_TO_SENSITIVITY)
    anchor = {
        "source": "Azure Public Dataset v1, vmtable (Cortez et al., SOSP'17)",
        "parametric_fallback": False,
        "url": VMTABLE_URL,
        "sha256": _sha256(DATA_PATH),
        "rows_total": len(df),
        "rows_used": len(used),
        "drop_fractions": {"created_at_trace_start": float(censored.mean())},
        "cores": {
            "values": CORE_VALUES,
            "probs": _probs(capped_cores.value_counts(), CORE_VALUES),
            "trace_counts": {str(k): int(v) for k, v in used.cores.value_counts().sort_index().items()},
            "capped_to_8_fraction": float((used.cores > max(CORE_VALUES)).mean()),
        },
        "sensitivity": {
            "probs": dict(zip(levels.SENSITIVITIES, _probs(sensitivity.value_counts(), levels.SENSITIVITIES))),
            "category_map": CATEGORY_TO_SENSITIVITY,
            "trace_counts": {k: int(v) for k, v in used.category.value_counts().items()},
            # Not used: includes the dropped t=0 VMs, a survivor-biased (long-lived) population.
            "probs_incl_trace_start": dict(zip(levels.SENSITIVITIES, _probs(
                df.category.map(CATEGORY_TO_SENSITIVITY).value_counts(), levels.SENSITIVITIES))),
        },
        # Cores and sensitivity are sampled independently; this is the association that discards.
        "cores_category_cramers_v": _cramers_v(capped_cores, used.category),
        "intensity": {
            "bin_seconds": BIN_SECONDS,
            "days": int((used.created // BIN_SECONDS).max() + 1) // levels.SHAPE_BINS,
            "unaligned_timestamp_fraction": float((used.created % BIN_SECONDS != 0).mean()),
            "shape": _intensity_shape(used.created),
        },
    }
    core_probs = anchor["cores"]["probs"]
    anchor["derived"] = {
        "E_cores": float(np.dot(CORE_VALUES, core_probs)),
        "E_runtime_ref": float(sum(p * levels.expected_runtime_ref(c) for c, p in zip(CORE_VALUES, core_probs))),
        "w_bar": levels.w_bar(anchor),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(anchor, f, indent=1)
        f.write("\n")
    summary = {k: v for k, v in anchor.items() if k != "intensity"}
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
