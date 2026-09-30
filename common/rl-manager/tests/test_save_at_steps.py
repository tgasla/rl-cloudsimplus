"""SaveAtStepsCallback on a real, tiny PPO / MaskablePPO (no JVM): model_at_<k> is the policy
after the first update that consumed at least k env steps, and its sidecar records how many.
Run: python3 -m pytest common/rl-manager/tests/test_save_at_steps.py"""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks.best_on_val_callback import BestOnValCallback
from callbacks.save_at_steps_callback import SaveAtStepsCallback
from utils import misc

N_ENVS, N_STEPS = 2, 8            # one rollout = 16 env steps
UPDATES_TO_64 = [0, 16, 32, 48, 64]


@pytest.fixture(autouse=True)
def _one_torch_thread():
    """A tiny network gains nothing from threads, and on a loaded machine they cost seconds."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class ToyEnv(gym.Env):
    """Guess the sign of the first observation coordinate; five steps per episode."""
    observation_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._obs = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        return self._obs, {}

    def step(self, action):
        reward = float((self._obs[0] > 0) == bool(action))
        self._t += 1
        self._obs = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        return self._obs, reward, self._t >= 5, False, {}

    def action_masks(self):
        return np.ones(2, dtype=bool)


def _params(policy) -> dict:
    return {name: t.detach().clone() for name, t in policy.state_dict().items()}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _model(algorithm):
    """`algorithm`, keeping its parameters after every train() under the env steps it consumed:
    the reference each model_at_<k> is compared against."""
    class Recording(algorithm):
        def train(self):
            super().train()
            self.after_train[self.num_timesteps] = _params(self.policy)

    model = Recording("MlpPolicy", DummyVecEnv([ToyEnv] * N_ENVS), n_steps=N_STEPS, batch_size=8,
                      n_epochs=1, learning_rate=1e-2, seed=0, device="cpu")
    model.after_train = {0: _params(model.policy)}
    return model


@pytest.mark.parametrize("algorithm", [PPO, MaskablePPO])
@pytest.mark.parametrize("total, steps", [(64, [0, 10, 16, 40, 64]), (60, [60])])
def test_model_at_k_is_the_policy_after_the_update_that_crossed_k(tmp_path, algorithm, total,
                                                                    steps):
    model = _model(algorithm)
    # wrapped as create_val_callback wraps it: the budgets check reads learn()'s locals through it
    model.learn(total, callback=CallbackList([SaveAtStepsCallback(str(tmp_path), steps)]))

    updates = sorted(model.after_train)
    assert updates == UPDATES_TO_64      # learn() overshoots 60 to the end of a whole rollout
    for before, after in zip(updates, updates[1:]):     # so a match pins down one update
        assert not _same(model.after_train[before], model.after_train[after])
    for k in steps:
        crossed = next(n for n in updates if n >= k)
        saved = algorithm.load(tmp_path / f"model_at_{k}", device="cpu")
        assert _same(_params(saved.policy), model.after_train[crossed]), k
        with open(tmp_path / f"model_at_{k}.json") as f:
            assert json.load(f) == {"k": k, "trained_steps": crossed}


class _StopAt(BaseCallback):
    """Stops training when num_timesteps reaches `at`, in the middle of a rollout."""

    def __init__(self, at: int):
        super().__init__()
        self.at = at

    def _on_step(self) -> bool:
        return self.num_timesteps < self.at


def test_a_rollout_cut_short_is_not_counted_as_trained(tmp_path, capsys):
    """Stopped at 40 env steps, inside the third rollout: the policy was updated after 16 and 32
    steps only, so model_at_20 is the policy after the update at 32, and nothing was ever
    trained on 40 steps."""
    model = _model(PPO)
    model.learn(64, callback=CallbackList([SaveAtStepsCallback(str(tmp_path), [20, 40]),
                                           _StopAt(40)]))
    assert model.num_timesteps == 40 and sorted(model.after_train) == [0, 16, 32]
    saved = PPO.load(tmp_path / "model_at_20", device="cpu")
    assert _same(_params(saved.policy), model.after_train[32])
    with open(tmp_path / "model_at_20.json") as f:
        assert json.load(f) == {"k": 20, "trained_steps": 32}
    assert not (tmp_path / "model_at_40.zip").exists()
    assert "stopped after 32 steps, model_at_[40] not saved" in capsys.readouterr().out


def test_a_budget_the_run_never_reaches_is_refused_before_training(tmp_path):
    model = _model(PPO)
    with pytest.raises(ValueError, match=r"\[48\] exceed this run's total_timesteps 32"):
        model.learn(32, callback=CallbackList([SaveAtStepsCallback(str(tmp_path), [16, 48])]))
    assert list(tmp_path.iterdir()) == []


def test_create_val_callback_adds_the_budgets_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(misc, "vectorize_env", lambda *a, **k: "val env")
    params = {"log_dir": str(tmp_path), "val_num_cpu": 2}

    callback, env = misc.create_val_callback(params, num_cpu=4)
    assert isinstance(callback, BestOnValCallback) and env == "val env"

    callback, _ = misc.create_val_callback({**params, "save_at_steps": [50000, 0, 5000]}, 4)
    assert isinstance(callback, CallbackList)
    best, budgets = callback.callbacks
    assert isinstance(best, BestOnValCallback) and isinstance(budgets, SaveAtStepsCallback)
    assert budgets.pending == [0, 5000, 50000]
