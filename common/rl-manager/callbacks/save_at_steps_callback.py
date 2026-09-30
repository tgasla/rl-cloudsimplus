import json
import os

from stable_baselines3.common.callbacks import BaseCallback


class SaveAtStepsCallback(BaseCallback):
    """Saves model_at_<k> for each budget k in `steps` (few-shot checkpoints: FS_k is an
    evaluation of such a checkpoint, never a point on the training curve).

    model_at_<k> holds the policy after the first update that has consumed at least k env
    steps, and model_at_<k>.json records how many it consumed ({"k", "trained_steps"}). The
    policy only changes in train(), after a whole rollout (n_envs * n_steps env steps), so a
    save from _on_step would hold a policy trained on fewer than k steps. The first hooks after
    train() are the next _on_rollout_start and, after the last update, _on_training_end; learn()
    runs until a full rollout reaches total_timesteps, so k == total_timesteps is saved there.
    """

    def __init__(self, log_dir: str, steps: list[int], verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.pending = sorted({int(k) for k in steps})
        self._trained = 0   # env steps behind the current parameters

    def _on_training_start(self) -> None:
        total = self.locals["total_timesteps"]
        beyond = [k for k in self.pending if k > total]
        if beyond:
            raise ValueError(f"save_at_steps {beyond} exceed this run's total_timesteps {total}")
        self._trained = self.model.num_timesteps

    def _on_rollout_end(self) -> None:
        self._trained = self.model.num_timesteps   # learn() calls train() on this rollout next

    def _on_rollout_start(self) -> None:
        self._save_due()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self._save_due()
        if self.pending:
            print(f"SaveAtStepsCallback: training stopped after {self._trained} steps, "
                  f"model_at_{self.pending} not saved")

    def _save_due(self) -> None:
        while self.pending and self.pending[0] <= self._trained:
            k = self.pending.pop(0)
            path = os.path.join(self.log_dir, f"model_at_{k}")
            self.model.save(path)
            with open(path + ".json", "w") as f:
                json.dump({"k": k, "trained_steps": int(self._trained)}, f)
