"""Write-once run directories.

A run directory is created exactly once. Re-running an experiment whose target
directory already holds a completed run is refused rather than merged into, so
results from two runs can never end up interleaved in one directory.

States:
    empty       target is absent or holds nothing -> proceed
    incomplete  a previous run died before finishing -> archived automatically
    completed   a previous run finished -> needs on_exists: archive
    legacy      non-empty, predates run_status.json -> needs on_exists: archive

The canonical path stays base_log_dir/experiment_dir/experiment_name; only the
displaced copy gets a timestamp, under base_log_dir/_archive/.
"""

import json
import os
from datetime import datetime, timezone

ARCHIVE_DIRNAME = "_archive"
STATUS_FILENAME = "run_status.json"
SKIP_EXIT_CODE = 3

_VALID_ON_EXISTS = ("abort", "archive")


class RunDirExistsError(Exception):
    """Target run directory already holds a run that will not be overwritten."""


def resolve_log_dir(params) -> str:
    """Build the canonical run directory and refuse anything outside base_log_dir."""
    base = str(params["base_log_dir"])
    experiment_dir = str(params.get("experiment_dir", "")).strip()
    experiment_name = str(params.get("experiment_name", "")).strip()

    if not experiment_dir or not experiment_name:
        raise ValueError(
            "experiment_dir and experiment_name must both be non-empty "
            f"(got experiment_dir={experiment_dir!r}, experiment_name={experiment_name!r})"
        )
    if experiment_dir.split(os.sep)[0] == ARCHIVE_DIRNAME:
        raise ValueError(f"experiment_dir must not start with {ARCHIVE_DIRNAME!r}")

    log_dir = os.path.join(base, experiment_dir, experiment_name)

    base_abs = os.path.abspath(base)
    log_abs = os.path.abspath(log_dir)
    if log_abs == base_abs or os.path.commonpath([base_abs, log_abs]) != base_abs:
        raise ValueError(f"resolved log_dir {log_abs!r} escapes base_log_dir {base_abs!r}")

    return log_dir


def classify(log_dir: str) -> str:
    """Return one of: empty, incomplete, completed, legacy."""
    if not os.path.isdir(log_dir):
        return "empty"

    entries = [e for e in os.listdir(log_dir) if not e.startswith(".")]
    if not entries:
        return "empty"

    if STATUS_FILENAME in entries:
        try:
            with open(os.path.join(log_dir, STATUS_FILENAME)) as fh:
                status = json.load(fh).get("status")
        except (OSError, ValueError, AttributeError):
            return "incomplete"
        return "completed" if status == "completed" else "incomplete"

    if STATUS_FILENAME + ".tmp" in entries:
        return "incomplete"   # died mid-write of the status file

    return "legacy"


def archive(params, log_dir: str) -> str:
    """Move log_dir aside under base_log_dir/_archive/... and return the new path.

    Stamped with the archiving time, not the run's finish time: file mtimes in a
    run directory are not a reliable record of when the run actually ended.
    """
    base = str(params["base_log_dir"])
    experiment_dir = str(params["experiment_dir"]).strip()
    experiment_name = str(params["experiment_name"]).strip()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = os.path.join(base, ARCHIVE_DIRNAME, experiment_dir, experiment_name)
    os.makedirs(parent, exist_ok=True)

    dest = os.path.join(parent, stamp)
    suffix = 1
    while os.path.exists(dest):
        dest = os.path.join(parent, f"{stamp}-{suffix}")
        suffix += 1

    os.rename(log_dir, dest)

    # TensorBoard finds runs by the "tfevents" substring; renaming keeps every
    # byte but stops archived runs from cluttering the run selector forever.
    for name in os.listdir(dest):
        if "tfevents" in name:
            os.rename(
                os.path.join(dest, name),
                os.path.join(dest, name.replace("tfevents", "tfarchived")),
            )

    return dest


def prepare_run_dir(params) -> str:
    """Resolve, guard and create this run's directory. Never writes into an existing run."""
    log_dir = resolve_log_dir(params)
    state = classify(log_dir)

    on_exists = str(params.get("on_exists", "abort")).strip().lower()
    if on_exists not in _VALID_ON_EXISTS:
        raise ValueError(
            f"on_exists must be one of {_VALID_ON_EXISTS}, got {on_exists!r}"
        )

    if state == "incomplete":
        dest = archive(params, log_dir)
        print(f"[run_dir] previous run at {log_dir} did not finish — archived to {dest}")
    elif state in ("completed", "legacy"):
        if on_exists != "archive":
            raise RunDirExistsError(
                f"REFUSED: {log_dir} already holds a {state} run.\n"
                f"         Add 'on_exists: archive' to this experiment to move it to "
                f"{os.path.join(str(params['base_log_dir']), ARCHIVE_DIRNAME)}/, "
                f"or delete the directory."
            )
        dest = archive(params, log_dir)
        print(f"[run_dir] on_exists=archive — previous {state} run moved to {dest}")

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)  # deliberately no exist_ok: a run dir is written once
    return log_dir


def write_run_status(log_dir: str, params, status: str) -> None:
    """Record this run's status atomically. Written first, so a crash leaves it 'running'."""
    payload = {
        "status": status,
        "mode": params.get("mode"),
        "experiment_dir": params.get("experiment_dir"),
        "experiment_name": params.get("experiment_name"),
        "feature_extractor": params.get("feature_extractor"),
        "timesteps": params.get("timesteps"),
        "seed": params.get("seed"),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = os.path.join(log_dir, STATUS_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)
