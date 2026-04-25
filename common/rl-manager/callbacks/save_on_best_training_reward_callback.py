import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Union
import torch
from collections import deque

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Unified callback for both SingleDC (main) and MultiDC (euromlsys) environments.
    Dynamically tracks all info keys returned by the environment — no hardcoded metric names.

    :param log_dir: Path to the folder where the model will be saved.
            It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        log_dir: str,
        save_replay_buffer: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_replay_buffer = save_replay_buffer
        self.model_save_path = os.path.join(log_dir, "best_model")
        self.best_reward = -np.inf
        self.previous_best_episode_num = None
        self.current_episode_num = 0
        self.best_episode_filename_prefix = "best_episode"

        # All tracked metric deques — dynamically populated from info keys
        self.metric_deques: Dict[str, deque] = {}
        # All tracked info lists per episode
        self.info_tracked_lists: Dict[str, list] = {}
        # Rolling episode lists
        self.observations = []
        self.actions = []
        self.rewards = []
        self.new_observations = []
        self.current_episode_length = 0

    def get(self, attr) -> Any:
        return self.training_env.env_method("get_wrapper_attr", attr)[0]

    def mean_exclude_neg(self, arr):
        non_negative_values = [value for value in arr if value >= 0]
        return np.mean(non_negative_values) if non_negative_values else 0.0

    def mean_of_non_empty_sublists(self, arr):
        non_empty_values = [value for sublist in arr if sublist for value in sublist]
        return np.mean(non_empty_values) if non_empty_values else 0.0

    def _create_episode_details_dict(self) -> Dict:
        timesteps = np.arange(
            self.num_timesteps - self.current_episode_length + 1, self.num_timesteps + 1
        )
        episode_details = {
            "timestep": timesteps,
            "reward": self.rewards,
        }
        # Add all dynamically tracked info lists
        for key, values in self.info_tracked_lists.items():
            episode_details[key] = values
        return episode_details

    def _clear_episode_details(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.new_observations = []
        self.current_episode_length = 0
        # Clear per-episode info lists
        for key in self.info_tracked_lists:
            self.info_tracked_lists[key] = []

    def _delete_previous_best(self) -> None:
        if self.previous_best_episode_num:
            log_file = os.path.join(
                self.log_dir,
                f"{self.best_episode_filename_prefix}_{self.previous_best_episode_num}.csv",
            )
            if os.path.exists(log_file):
                os.remove(log_file)

    def _get_observation_from_locals(self, obs_key) -> Union[list, dict]:
        obs = self.locals[obs_key]
        if isinstance(obs, dict):
            processed_obs = {}
            for key, value in obs.items():
                value = value[0].cpu() if isinstance(value, torch.Tensor) else value[0]
                processed_obs[key] = value
        elif isinstance(obs, torch.Tensor):
            processed_obs = obs[0].cpu().numpy()
        elif isinstance(obs, np.ndarray):
            processed_obs = obs[0]
        else:
            raise TypeError(f"Unknown observation type: {type(obs)}")
        return processed_obs

    def _save_timestep_details(self) -> None:
        self.observations.append(self._get_observation_from_locals("obs_tensor"))
        self.actions.append(self.locals["actions"][0])
        self.rewards.append(self.locals["rewards"][0])
        self.new_observations.append(self._get_observation_from_locals("new_obs"))

        info = self.locals["infos"][0]

        # Dynamically discover and track all numeric info keys
        for key, value in info.items():
            # Skip non-numeric/non-serializable values
            if isinstance(value, (list, dict)):
                continue
            # Initialize tracking on first encounter
            if key not in self.info_tracked_lists:
                self.info_tracked_lists[key] = []
                self.metric_deques[key] = deque(maxlen=100)
            self.info_tracked_lists[key].append(value)

    def _maybe_save_replay_buffer(self) -> None:
        if (
            hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
            and self.save_replay_buffer
        ):
            replay_buffer_path = os.path.join(self.log_dir, "best_model_replay_buffer")
            if self.verbose >= 1:
                print(f"Saving replay buffer to {replay_buffer_path}")
            self.model.save_replay_buffer(replay_buffer_path)

    def _create_csv_paths(self) -> Dict:
        paths = {
            "actions": os.path.join(self.log_dir, "actions.csv"),
            "episode_details": os.path.join(
                self.log_dir,
                f"{self.best_episode_filename_prefix}_{self.current_episode_num}.csv",
            ),
        }
        # Add a path for each tracked info key that is a list of numbers
        for key in self.info_tracked_lists:
            if key != "observation_tree_array":  # skip large arrays
                paths[f"info_{key}"] = os.path.join(self.log_dir, f"{key}.csv")
        return paths

    def _create_dataframes(self) -> Dict:
        df_dict = {
            "episode_actions": pd.DataFrame(self.actions),
            "episode_details": pd.DataFrame(self._create_episode_details_dict()),
        }
        # Create dataframe for each tracked info key
        for key, values in self.info_tracked_lists.items():
            df_dict[f"info_{key}"] = pd.DataFrame({key: values})
        return df_dict

    def _write_dataframes_to_csvs(self, df_dict, path_dict) -> None:
        df_dict["episode_actions"].to_csv(
            path_dict["actions"],
            index=False,
            header=False,
        )
        df_dict["episode_details"].to_csv(path_dict["episode_details"], index=False)
        # Write each info key dataframe
        for key in self.info_tracked_lists:
            if key in path_dict:
                df_dict[f"info_{key}"].to_csv(path_dict[f"info_{key}"], header=False)

    def _maybe_save_best_episode_details(self) -> None:
        self._delete_previous_best()
        self.previous_best_episode_num = self.current_episode_num

        path_dict = self._create_csv_paths()

        if self.verbose >= 1:
            print(f"Saving simulation details to {path_dict}")

        df_dict = self._create_dataframes()
        self._write_dataframes_to_csvs(df_dict, path_dict)

    def _maybe_print_current_episode_info(self, current_reward) -> None:
        if self.verbose >= 1:
            print(f"Current timesteps: {self.num_timesteps - 1}")
            print(f"Episode number: {self.current_episode_num}")
            print(f"Episode length: {len(self.observations)}")
            print(f"Best reward: {self.best_reward:.2f}")
            print(f"Current reward: {current_reward:.2f}")

    def _save_new_best(self) -> None:
        if self.verbose >= 1:
            print(f"Saving new best model to {self.model_save_path}")
        self.model.save(self.model_save_path)
        self._maybe_save_replay_buffer()
        self._maybe_save_best_episode_details()

    def _write_progress_log_row(self) -> None:
        ep_first_timestep = self.num_timesteps - self.current_episode_length + 1
        ep_last_timestep = self.num_timesteps
        self.logger.record(
            "train/episode_num", self.current_episode_num, exclude="tensorboard"
        )
        self.logger.record(
            "train/episode_length", self.current_episode_length, exclude="tensorboard"
        )
        self.logger.record(
            "train/ep_first_timestep", ep_first_timestep, exclude="tensorboard"
        )
        self.logger.record(
            "train/ep_last_timestep", ep_last_timestep, exclude="tensorboard"
        )
        self.logger.record(
            "train/ep_total_rew", np.sum(self.rewards), exclude="tensorboard"
        )

        # Compute and record rolling averages for all tracked info keys
        for key, values in self.info_tracked_lists.items():
            # Choose appropriate aggregation method
            if all(isinstance(v, (int, float)) and v >= 0 for v in values):
                # Non-negative metrics (like ratios): exclude -1 sentinel values
                deque_val = self.mean_exclude_neg(values)
            else:
                # Rewards and signed metrics: use regular mean
                deque_val = np.mean(values) if values else 0.0
            self.metric_deques[key].append(deque_val)
            # Record per-episode aggregate
            self.logger.record(f"train/ep_{key}", deque_val, exclude="tensorboard")

        self.logger.dump()

    def _on_step(self) -> bool:
        self.current_episode_length += 1
        self._save_timestep_details()

        if self.locals["dones"][0]:
            self.current_episode_num += 1
            if self.verbose >= 1:
                print("Episode terminated")
            self._write_progress_log_row()

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) == 0:
                return True
            current_reward = float(y[-1])
            self._maybe_print_current_episode_info(current_reward)

            # New best model, save the agent and the episode details
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self._save_new_best()

            self._clear_episode_details()
        return True

    def _on_training_start(self):
        super()._on_training_start()

    def _on_rollout_end(self):
        super()._on_rollout_end()
        # Record rolling mean for every tracked metric
        for key, deque_obj in self.metric_deques.items():
            mean_val = np.mean(deque_obj) if deque_obj else 0.0
            self.logger.record(f"rollout/ep_{key}_mean", mean_val)

        self.logger.dump(step=self.num_timesteps)