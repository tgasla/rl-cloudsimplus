import os
from collections import deque

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from utils.evaluation import model_predictor, play_levels


class BestOnValCallback(BaseCallback):
    """Checkpoint selection on held-out levels, not on the training curve.

    Every `every` environment steps, and once more when training ends, the deterministic
    policy plays each level of the val split once on `eval_env`. The checkpoint with the
    highest mean unshaped return is kept as best_val_model and the last one as final_model.
    Every sweep appends one row per level to val.csv, so the selection can be audited.

    It also logs the unshaped return of finished training episodes (rollout/ep_unshaped_mean):
    SB3's ep_rew_mean sums the reward the agent learns from, which includes shaping when it
    is on, while the benchmark metric does not.
    """

    def __init__(self, eval_env, quotas: list[int], log_dir: str, every: int, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.quotas = quotas
        self.log_dir = log_dir
        self.every = every
        self.best_mean = -np.inf
        self._last_sweep = 0
        self._episode_unshaped = None
        self._finished_unshaped = deque(maxlen=100)

    def _on_step(self) -> bool:
        if self._episode_unshaped is None:
            self._episode_unshaped = np.zeros(self.training_env.num_envs)
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            self._episode_unshaped[i] += info["unshaped_reward"]
            if done:
                self._finished_unshaped.append(self._episode_unshaped[i])
                self._episode_unshaped[i] = 0.0
        if self._finished_unshaped:
            self.logger.record("rollout/ep_unshaped_mean", float(np.mean(self._finished_unshaped)))
        if self.num_timesteps - self._last_sweep >= self.every:
            self._sweep()
        return True

    def _on_training_end(self) -> None:
        if self._last_sweep != self.num_timesteps:
            self._sweep()
        self.model.save(os.path.join(self.log_dir, "final_model"))

    def _sweep(self) -> None:
        self._last_sweep = self.num_timesteps
        rows = play_levels(self.eval_env, model_predictor(self.model), self.quotas)
        mean = float(np.mean([row["unshaped_return"] for row in rows]))
        path = os.path.join(self.log_dir, "val.csv")
        pd.DataFrame(rows).assign(timestep=self.num_timesteps).to_csv(
            path, mode="a", header=not os.path.exists(path), index=False)
        self.logger.record("val/mean_unshaped_return", mean)
        if mean > self.best_mean:
            self.best_mean = mean
            self.model.save(os.path.join(self.log_dir, "best_val_model"))
            if self.verbose:
                print(f"val {mean:.4f} at {self.num_timesteps}: new best_val_model")
