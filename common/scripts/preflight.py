#!/usr/bin/env python3
"""Host-side preflight for an experiment queue.

Prints what `make run` is about to do before the first container starts, and
blocks on exactly one condition: a transfer/test whose source model neither
exists on disk nor is produced by an earlier experiment in the same queue.

Everything else is a warning. Occupied targets are reported as SKIP and the
queue still runs — restarting a partially-finished queue is a normal operation,
not an error. Any internal failure here fails open (exit 0): a bug in preflight
must never stop a twenty-hour queue.

Deliberately duplicates two trivial helpers from utils/run_dir.py rather than
importing container code, so an ImportError in the container tree cannot block
`make run`.

Usage: preflight.py <config.yml> <logs_root>
"""

import json
import os
import sys

STATUS_FILENAME = "run_status.json"
MODEL_FILENAME = "best_model.zip"
PRODUCING_MODES = ("train", "transfer")


def classify(run_dir):
    if not os.path.isdir(run_dir):
        return "empty"
    entries = [e for e in os.listdir(run_dir) if not e.startswith(".")]
    if not entries:
        return "empty"
    if STATUS_FILENAME in entries:
        try:
            with open(os.path.join(run_dir, STATUS_FILENAME)) as fh:
                return "completed" if json.load(fh).get("status") == "completed" else "incomplete"
        except (OSError, ValueError, AttributeError):
            return "incomplete"
    if STATUS_FILENAME + ".tmp" in entries:
        return "incomplete"   # died mid-write of the status file
    return "legacy"


def load_experiments(config_path):
    import yaml

    # The real !include resolves relative to the domain dir inside the container;
    # topologies/ does not exist at that path on the host. Stub every custom tag.
    yaml.add_multi_constructor("!", lambda loader, suffix, node: None, Loader=yaml.SafeLoader)
    with open(config_path) as fh:
        cfg = yaml.load(fh, Loader=yaml.SafeLoader)
    common = cfg.get("common") or {}
    return [{**common, **(e or {})} for e in (cfg.get("experiments") or [])]


def run(config_path, logs_root):
    experiments = load_experiments(config_path)
    root_base = os.path.basename(os.path.normpath(logs_root))
    print(f"\n── preflight: {config_path} ({len(experiments)} experiments) "
          + "─" * max(0, 30 - len(config_path)))

    if not experiments:
        print("no experiments enabled — nothing to run.")
        return 0

    producers = {}       # "<experiment_dir>/<experiment_name>" -> (index, will_skip)
    seen_targets = {}    # duplicate detection within this queue
    errors, warnings = [], []
    skip_count = 0

    print(f" {'#':>2}  {'mode':<9} {'target':<62} state")
    for idx, exp in enumerate(experiments, 1):
        mode = str(exp.get("mode", "?"))
        edir = str(exp.get("experiment_dir", "")).strip()
        ename = str(exp.get("experiment_name", "")).strip()

        if not edir or not ename:
            errors.append(f"#{idx} ({mode}) has an empty experiment_dir or experiment_name")
            continue

        # base_log_dir is queue-level: it sets the container bind mount and this root.
        # A per-experiment override would write somewhere neither of them points at.
        exp_base = str(exp.get("base_log_dir", root_base))
        if exp_base != root_base:
            errors.append(
                f"#{idx} overrides base_log_dir to {exp_base!r}, but the run is mounted at "
                f"{root_base!r}. base_log_dir must be set once under common:."
            )
            continue

        target = f"{edir}/{ename}"
        state = classify(os.path.join(logs_root, edir, ename))
        on_exists = str(exp.get("on_exists", "abort")).strip().lower()
        will_skip = state in ("completed", "legacy") and on_exists != "archive"

        if target in seen_targets:
            # Once the earlier experiment has run, this target is occupied whatever
            # is on disk now, so the outcome depends only on this entry's on_exists.
            will_skip = on_exists != "archive"
            consequence = ("the later one will ARCHIVE the earlier one's output"
                           if on_exists == "archive"
                           else "the later one will be SKIPPED, the earlier one's output kept")
            warnings.append(
                f"#{idx} writes to the same target as #{seen_targets[target]} ({target}) — "
                f"{consequence}"
            )
        seen_targets[target] = idx

        note = ""
        if mode in ("transfer", "test"):
            src = str(exp.get("train_model_dir", "")).strip()
            if not src:
                errors.append(f"#{idx} ({mode}) has no train_model_dir")
                continue
            src_on_disk = os.path.exists(os.path.join(logs_root, src, MODEL_FILENAME))
            produced = producers.get(src)
            if not src_on_disk and produced is None:
                errors.append(
                    f"#{idx} ({mode}) source {src} has no {MODEL_FILENAME} on disk and is not "
                    f"produced by an earlier experiment in this queue"
                )
                continue
            if produced is not None:
                if produced[1] and not src_on_disk:
                    errors.append(
                        f"#{idx} ({mode}) source {src} has no {MODEL_FILENAME} on disk and its "
                        f"in-queue producer #{produced[0]} will itself be skipped — add "
                        f"'on_exists: archive' to #{produced[0]}, or delete its run directory"
                    )
                    continue
                note = f"<- #{produced[0]}" if not src_on_disk else f"<- #{produced[0]} (model on disk)"
                if produced[1] and not will_skip:
                    warnings.append(
                        f"#{idx} will fine-tune the EXISTING model at {src} because its "
                        f"producer #{produced[0]} will be skipped"
                    )
            else:
                note = "<- on disk"

        if mode in PRODUCING_MODES:
            producers[target] = (idx, will_skip)

        if will_skip:
            skip_count += 1
        status = "SKIP (target holds a %s run)" % state if will_skip else (
            "ok (deferred)" if note.startswith("<- #") else "ok")
        print(f" {idx:>2}  {mode:<9} {target:<62} {status} {note}".rstrip())

    for w in warnings:
        print(f" WARN {w}")
    for e in errors:
        print(f" ERROR {e}")

    if errors:
        print(f"\npreflight FAILED — {len(errors)} experiment(s) cannot run. Nothing started.\n")
        return 1

    print(f"\npreflight OK — {len(experiments)} experiments queued"
          + (f", {skip_count} will be skipped.\n" if skip_count else ".\n"))
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print(f"usage: {sys.argv[0]} <config.yml> <logs_root>", file=sys.stderr)
            sys.exit(0)  # fail open
        sys.exit(run(sys.argv[1], sys.argv[2]))
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - fail open, never block the queue
        print(f"preflight skipped (internal error: {type(exc).__name__}: {exc})")
        sys.exit(0)
