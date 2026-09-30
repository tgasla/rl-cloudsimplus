"""evaluate.py provenance: which policy an evaluation row belongs to, read from the
run_status.json files the entrypoint writes, for every kind of run the analysis tells apart.
Run: python3 -m pytest common/rl-manager/tests/test_provenance.py"""

import json
import os

import pandas as pd
import pytest

import evaluate
from utils import misc
from utils.levels import TEST_LEVELS, load_topology
from utils.run_dir import write_run_status

SOURCE = "src/a5_s1234"
RING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "topologies", "ring")


def _run(logs, run_dir, status="completed", **params):
    """A run directory with the run_status.json the entrypoint writes for these params."""
    os.makedirs(logs / run_dir)
    write_run_status(str(logs / run_dir), {"feature_extractor": "a5", "timesteps": 50000,
                                           "cloudlet_to_dc_mapping": "rl", **params}, status)
    return run_dir


def _sidecar(logs, run_dir, k, trained_steps):
    with open(logs / run_dir / f"model_at_{k}.json", "w") as f:
        json.dump({"k": k, "trained_steps": trained_steps}, f)


def _provenance(logs, **params):
    return evaluate.provenance({"base_log_dir": str(logs), "cloudlet_to_dc_mapping": "rl",
                                "benchmark_member": "C1-N7", **params})


@pytest.fixture
def logs(tmp_path):
    logs = tmp_path / "logs"
    _run(logs, SOURCE, mode="train", benchmark_member="S", seed=1234, timesteps=600000)
    return logs


def _expected(**fields):
    return {**dict.fromkeys(evaluate.PROVENANCE), "arch": "a5", "run_status": "completed",
            **fields}


@pytest.mark.parametrize("checkpoint", [None, "best_val_model", "final_model"])
def test_a_source_run_zero_shot(logs, checkpoint):
    assert _provenance(logs, train_model_dir=SOURCE, checkpoint=checkpoint) == _expected(
        run=SOURCE, checkpoint=checkpoint or "best_val_model", trained_on="S", source="S",
        source_run=SOURCE, seed=1234)


def test_a_run_trained_from_scratch_on_the_target_at_a_budget(logs):
    run = _run(logs, "scratch/a5_c1n7", mode="train", benchmark_member="C1-N7", seed=2345)
    _sidecar(logs, run, 5000, 6144)
    assert _provenance(logs, train_model_dir=run, checkpoint="model_at_5000") == _expected(
        run=run, checkpoint="model_at_5000", trained_steps=6144, trained_on="C1-N7",
        source="C1-N7", source_run=run, seed=2345)


@pytest.mark.parametrize("scope, recorded", [(None, "full"), ("full", "full"), ("head", "head"),
                                             ("extractor", "extractor")])
def test_a_fine_tuned_run_at_each_scope(logs, scope, recorded):
    run = _run(logs, f"ft/{recorded}", mode="transfer", benchmark_member="C1-N7",
               train_model_dir=SOURCE, finetune=scope, seed=7)
    _sidecar(logs, run, 20000, 20480)
    assert _provenance(logs, train_model_dir=run, checkpoint="model_at_20000") == _expected(
        run=run, checkpoint="best_val_model>model_at_20000", trained_steps=20480,
        trained_on="C1-N7", finetune=recorded, source="S", source_run=SOURCE, seed=1234)
    assert _provenance(logs, train_model_dir=run, checkpoint="final_model")["checkpoint"] == \
        "best_val_model>final_model"


def test_a_run_fine_tuned_from_the_sources_final_model(logs):
    run = _run(logs, "ft/from_final", mode="transfer", benchmark_member="GAM-lo",
               train_model_dir=SOURCE, checkpoint="final_model")
    assert _provenance(logs, train_model_dir=run)["checkpoint"] == "final_model>best_val_model"


def test_a_nested_transfer_chain_names_every_link(logs):
    """S -> C1-N7 (head, from best_val_model) -> C1-N19 (full, from that run's final_model)."""
    first = _run(logs, "chain/c1n7", mode="transfer", benchmark_member="C1-N7",
                 train_model_dir=SOURCE + "/", finetune="head", seed=5)
    second = _run(logs, "chain/c1n19", mode="transfer", benchmark_member="C1-N19",
                  train_model_dir=first, checkpoint="final_model", seed=6)
    _sidecar(logs, second, 5000, 6144)
    assert _provenance(logs, train_model_dir=second + "/", checkpoint="model_at_5000",
                       benchmark_member="C1-N19") == _expected(
        run=second, checkpoint="best_val_model>final_model>model_at_5000", trained_steps=6144,
        trained_on="C1-N19", finetune="full", source="S", source_run=SOURCE, seed=1234)


def test_a_run_on_sb3s_default_network_is_arch_default(logs):
    run = _run(logs, "src/default", mode="train", benchmark_member="S", seed=1,
               feature_extractor=None)
    assert _provenance(logs, train_model_dir=run)["arch"] == "default"


def test_a_rule_based_policy_has_no_lineage():
    assert evaluate.provenance({"cloudlet_to_dc_mapping": "earliest-shortest-to-most-free-dc"}) \
        == {**dict.fromkeys(evaluate.PROVENANCE), "arch": "earliest-shortest-to-most-free-dc",
            "checkpoint": "rule"}


def test_an_unfinished_run_is_recorded_as_such(logs):
    run = _run(logs, "src/crashed", status="failed", mode="train", benchmark_member="S", seed=1)
    assert _provenance(logs, train_model_dir=run)["run_status"] == "failed"


def test_a_run_fine_tuned_from_an_unfinished_source_is_not_a_result(logs):
    """A crashed source still leaves a best_val_model, so the transfer itself can finish."""
    _run(logs, "src/interrupted", status="interrupted", mode="train", benchmark_member="S")
    run = _run(logs, "ft/from_interrupted", mode="transfer", benchmark_member="C1-N7",
               train_model_dir="src/interrupted")
    assert _provenance(logs, train_model_dir=run)["run_status"] == "interrupted"


def test_a_budget_checkpoint_without_its_step_record_is_refused(logs):
    with pytest.raises(FileNotFoundError, match="model_at_5000.json"):
        _provenance(logs, train_model_dir=SOURCE, checkpoint="model_at_5000")


def test_a_chain_with_a_missing_link_is_refused(logs):
    run = _run(logs, "ft/orphan", mode="transfer", benchmark_member="C1-N7",
               train_model_dir="src/deleted")
    with pytest.raises(FileNotFoundError, match="src/deleted"):
        _provenance(logs, train_model_dir=run)


class _Env:
    def close(self):
        pass


def _stub_simulator(monkeypatch, loaded):
    monkeypatch.setattr(evaluate, "vectorize_env", lambda *a, **k: _Env())
    monkeypatch.setattr(evaluate, "get_suitable_device", lambda name: "cpu")
    monkeypatch.setattr(evaluate, "play_levels", lambda env, predict, quotas: [
        {"level_id": level, "unshaped_return": 0.5} for level in reversed(TEST_LEVELS)])

    class Algorithm:
        @staticmethod
        def load(path, env, device):
            loaded.append(path)
            return "model"
    monkeypatch.setattr(evaluate, "get_algorithm", lambda name, params: Algorithm)


@pytest.mark.parametrize("mapping", ["rl", "earliest-most-critical-to-nearest-dc"])
def test_evaluate_stamps_every_row_with_the_provenance(monkeypatch, tmp_path, logs, mapping):
    loaded = []
    _stub_simulator(monkeypatch, loaded)
    out = tmp_path / "eval"
    out.mkdir()
    params = {"base_log_dir": str(logs), "cloudlet_to_dc_mapping": mapping, "level_split": "test",
              "benchmark_member": "C1-N7", "num_cpu": 4, "rl_algorithm": "MaskablePPO",
              "max_jobs_waiting": 32, "log_dir": str(out), "train_model_dir": SOURCE,
              "checkpoint": "final_model"}
    evaluate.evaluate(params, [])

    rows = pd.read_csv(out / "evaluation.csv")
    assert list(rows["level_id"]) == list(TEST_LEVELS)
    assert set(evaluate.PROVENANCE) <= set(rows.columns)
    stamped = rows[list(evaluate.PROVENANCE)].drop_duplicates()
    assert len(stamped) == 1
    expected = evaluate.provenance(params)
    assert {k: (None if pd.isna(v) else v) for k, v in stamped.iloc[0].items()} == expected
    assert loaded == ([os.path.join(str(logs), SOURCE, "final_model")] if mapping == "rl" else [])


class _LevelWorkers:
    """A vectorised env as far as the levels go: worker `rank` plays what misc.level_stream
    deals it from the params the env is built with, as in _create_grpc_env_for_rank."""

    def __init__(self, num_cpu, params):
        self.samplers = [misc.level_stream(params, rank)[1] for rank in range(num_cpu)]

    def close(self):
        pass


def _play_dealt_levels(env, predict, quotas):
    """play_levels: worker i plays quotas[i] episodes, each on its sampler's next level."""
    return [{"level_id": env.samplers[i].next(), "unshaped_return": 0.5}
            for i, quota in enumerate(quotas) for _ in range(quota)]


def _rule_params(tmp_path) -> dict:
    """An evaluate experiment of a rule on member S's test split, with no num_cpu anywhere."""
    out = tmp_path / "eval"
    out.mkdir()
    return {"base_log_dir": str(tmp_path), "log_dir": str(out), "level_split": "test",
            "cloudlet_to_dc_mapping": "earliest-shortest-to-most-free-dc",
            "rl_algorithm": "MaskablePPO", "max_jobs_waiting": 32, "benchmark_member": "S",
            "ring_manifest": os.path.join(RING, "manifest.json"),
            "datacenters": load_topology(os.path.join(RING, "S.yml")),
            "max_episode_length": 200, "timestep_interval": 1.0, "seed": 1234}


def test_evaluate_without_num_cpu_plays_every_level_once(monkeypatch, tmp_path):
    """evaluate plays on 16 workers by default and level_stream deals 1 worker's share by
    default: with the env params not stating 16, 16 workers each started a full pass of the
    split at their own rank and played 48 episodes on 18 distinct levels."""
    monkeypatch.setattr(evaluate, "vectorize_env", lambda env, algorithm, num_cpu, params,
                        jobs_json: _LevelWorkers(num_cpu, params))
    monkeypatch.setattr(evaluate, "play_levels", _play_dealt_levels)
    params = _rule_params(tmp_path)
    assert "num_cpu" not in params
    evaluate.evaluate(params, [])
    rows = pd.read_csv(tmp_path / "eval" / "evaluation.csv")
    assert rows["level_id"].tolist() == list(TEST_LEVELS)


def test_evaluate_refuses_an_env_that_does_not_play_the_split(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "vectorize_env", lambda *a, **k: _Env())
    monkeypatch.setattr(evaluate, "play_levels", lambda env, predict, quotas: [
        {"level_id": TEST_LEVELS[rank + i], "unshaped_return": 0.5}
        for rank in range(16) for i in range(3)])            # the 18-level replay of the defect
    with pytest.raises(RuntimeError, match="48 episodes on 18 distinct levels, not each of the "
                                           "48 test levels once"):
        evaluate.evaluate(_rule_params(tmp_path), [])
    assert os.listdir(tmp_path / "eval") == []


def test_evaluate_checks_the_chain_before_starting_the_simulators(monkeypatch, logs):
    def start(*args, **kwargs):
        raise AssertionError("JVMs started before the provenance was checked")
    monkeypatch.setattr(evaluate, "vectorize_env", start)
    with pytest.raises(FileNotFoundError):
        evaluate.evaluate({"base_log_dir": str(logs), "cloudlet_to_dc_mapping": "rl",
                           "level_split": "test", "benchmark_member": "S",
                           "rl_algorithm": "MaskablePPO", "train_model_dir": "src/missing"}, [])
