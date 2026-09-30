"""BestOnValCallback on a real, tiny MaskablePPO (no JVM): each val sweep scores the policy of
an update boundary under the env steps it was trained on, final_model is always scored, the
logged val points carry the same steps, and a sweep must play every val level exactly once.
Run: python3 -m pytest common/rl-manager/tests/test_best_on_val.py"""

import itertools

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

import callbacks.best_on_val_callback as best_on_val
from callbacks.best_on_val_callback import BestOnValCallback
from callbacks.save_at_steps_callback import SaveAtStepsCallback
from utils.evaluation import EPISODE_SUMS, level_quotas
from utils.levels import VAL_LEVELS

N_ENVS, N_STEPS = 2, 8            # one rollout = 16 env steps
TOTAL = 64
VAL_WORKERS = 2


@pytest.fixture(autouse=True)
def _one_torch_thread():
    """A tiny network gains nothing from threads, and on a loaded machine they cost seconds."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class LevelToy(gym.Env):
    """Name both digits of a context in 0..8; four steps per episode. Given `levels`, each reset
    plays the next of them (round-robin, as a val worker does), with contexts fixed per level."""
    observation_space = gym.spaces.Box(0.0, 1.0, (9,), np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3])
    params = {"datacenters": [{"name": "dc_b"}, {"name": "dc_a"}]}

    def __init__(self, seed=0, levels=None):
        self._rng = np.random.default_rng(seed)
        self._levels = itertools.cycle(levels) if levels else None
        self._level = None

    def _obs(self):
        self._context = int(self._rng.integers(9))
        return np.eye(9, dtype=np.float32)[self._context]

    def reset(self, *, seed=None, options=None):
        if self._levels is not None:
            self._level = next(self._levels)
            self._rng = np.random.default_rng(self._level)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        reward = float(action[0] == self._context % 3) + float(action[1] == self._context // 3)
        self._t += 1
        info = dict.fromkeys(EPISODE_SUMS, 0.0)
        info.update(unshaped_reward=reward, level_id=self._level, offered_value=2.0)
        return self._obs(), reward, self._t >= 4, False, info

    def action_masks(self):
        return np.ones(6, dtype=bool)


def _val_env(shares=None):
    shares = shares or [list(VAL_LEVELS)[rank::VAL_WORKERS] for rank in range(VAL_WORKERS)]
    return DummyVecEnv([lambda share=share: LevelToy(levels=share) for share in shares])


def _params(policy) -> dict:
    return {name: t.detach().clone() for name, t in policy.state_dict().items()}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class Recording(MaskablePPO):
    """Keeps its parameters after every train() under the env steps they were trained on."""

    def train(self):
        super().train()
        self.after_train[self.num_timesteps] = _params(self.policy)


def _train(tmp_path, monkeypatch, every, wrap=False):
    """Train TOTAL steps with the callback; return the model and the parameters each sweep
    scored, in order."""
    model = Recording("MlpPolicy", DummyVecEnv([lambda i=i: LevelToy(seed=i) for i in range(N_ENVS)]),
                      n_steps=N_STEPS, batch_size=N_ENVS * N_STEPS, n_epochs=1, learning_rate=1e-2,
                      seed=0, device="cpu")
    model.after_train = {0: _params(model.policy)}
    model.set_logger(configure(str(tmp_path), ["csv"]))
    scored, play = [], best_on_val.play_levels

    def spy(vec_env, predict, quotas):
        scored.append(_params(model.policy))
        return play(vec_env, predict, quotas)

    monkeypatch.setattr(best_on_val, "play_levels", spy)
    callback = BestOnValCallback(_val_env(), level_quotas("val", VAL_WORKERS), str(tmp_path), every)
    if wrap:        # as create_val_callback wraps it when save_at_steps is set
        callback = CallbackList([callback, SaveAtStepsCallback(str(tmp_path), [TOTAL])])
    model.learn(TOTAL, callback=callback)
    return model, scored


@pytest.mark.parametrize("wrap", [False, True], ids=["bare", "with_save_at_steps"])
@pytest.mark.parametrize("every, labels", [(16, [16, 32, 48, 64]), (20, [32, 64]),
                                           (8, [16, 32, 48, 64])],
                         ids=["every_rollout", "between_rollouts", "within_a_rollout"])
def test_each_sweep_scores_the_policy_trained_on_its_label(tmp_path, monkeypatch, every, labels,
                                                           wrap):
    model, scored = _train(tmp_path, monkeypatch, every, wrap)
    val = pd.read_csv(tmp_path / "val.csv")
    assert val["timestep"].unique().tolist() == labels
    for _, sweep in val.groupby("timestep"):
        assert sorted(sweep["level_id"]) == list(VAL_LEVELS)
    assert len(scored) == len(labels)
    for label, params in zip(labels, scored):
        assert _same(params, model.after_train[label]), label
    final = MaskablePPO.load(tmp_path / "final_model", device="cpu")
    assert _same(_params(final.policy), scored[-1])         # final_model was scored


def test_val_points_are_logged_at_their_sweeps(tmp_path, monkeypatch):
    _train(tmp_path, monkeypatch, every=20)
    means = pd.read_csv(tmp_path / "val.csv").groupby("timestep")["unshaped_return"].mean()
    logged = pd.read_csv(tmp_path / "progress.csv").dropna(subset=["val/mean_unshaped_return"])
    assert logged["time/total_timesteps"].tolist() == means.index.tolist() == [32, 64]
    np.testing.assert_allclose(logged["val/mean_unshaped_return"], means)


def test_a_sweep_that_does_not_play_each_val_level_once_is_refused(tmp_path):
    # Both workers play the first worker's share, as when the level samplers and the val env
    # disagree about the number of workers: 24 episodes, 12 distinct levels.
    model = MaskablePPO("MlpPolicy", DummyVecEnv([LevelToy]), n_steps=16, batch_size=16,
                        n_epochs=1, seed=0, device="cpu")
    first_share = list(VAL_LEVELS)[0::VAL_WORKERS]
    callback = BestOnValCallback(_val_env([first_share, first_share]),
                                 level_quotas("val", VAL_WORKERS), str(tmp_path), every=16)
    with pytest.raises(RuntimeError, match="24 episodes on 12 distinct levels, not each of the "
                                           "24 val levels once"):
        model.learn(32, callback=callback)
    assert not (tmp_path / "val.csv").exists()
