"""common/scripts/preflight.py: which saved model a queued run needs from its source."""

import os
import sys

import pytest

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "common", "scripts"))
import preflight  # noqa: E402


def _queue(tmp_path, experiments, common=""):
    config = tmp_path / "config.yml"
    config.write_text("common:\n  base_log_dir: logs\n" + common + "experiments:\n" + "".join(
        "  - " + "\n    ".join(f"{k}: {v}" for k, v in e.items()) + "\n" for e in experiments))
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return str(config), str(logs)


def _source(logs, name, *files):
    os.makedirs(os.path.join(logs, name), exist_ok=True)
    for f in files:
        open(os.path.join(logs, name, f), "w").close()


RING = "  benchmark_member: S\n"


@pytest.mark.parametrize("mode", ["transfer", "test", "evaluate"])
def test_ring_runs_need_the_sources_best_val_model(tmp_path, mode):
    config, logs = _queue(tmp_path, [dict(mode=mode, experiment_dir="d", experiment_name="t",
                                          train_model_dir="src/run")], common=RING)
    _source(logs, "src/run", "best_model.zip")
    assert preflight.run(config, logs) == 1
    _source(logs, "src/run", "best_val_model.zip")
    assert preflight.run(config, logs) == 0


def test_an_explicit_checkpoint_and_legacy_runs(tmp_path):
    config, logs = _queue(tmp_path, [
        dict(mode="transfer", experiment_dir="d", experiment_name="a", train_model_dir="src/run",
             checkpoint="final_model"),
        dict(mode="transfer", experiment_dir="d", experiment_name="b", train_model_dir="old/run"),
    ])
    _source(logs, "src/run", "final_model.zip")
    _source(logs, "old/run", "best_model.zip")
    assert preflight.run(config, logs) == 0


def test_evaluating_a_heuristic_needs_no_model(tmp_path):
    config, logs = _queue(tmp_path, [dict(
        mode="evaluate", experiment_dir="d", experiment_name="h",
        cloudlet_to_dc_mapping="earliest-shortest-to-most-free-dc")], common=RING)
    assert preflight.run(config, logs) == 0
