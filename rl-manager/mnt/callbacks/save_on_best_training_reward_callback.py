import csv
import os
import numpy as np
import pandas as pd

# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward.

    :param log_dir: Path to the folder where the model will be saved.
            It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        log_dir: str,
        save_replay_buffer: bool = True,
        save_best_episode_details: bool = True,
        verbose: int = 1,
    ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_replay_buffer = (save_replay_buffer,)
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_reward = -np.inf
        self.save_best_episode_details = save_best_episode_details
        self.previous_best_episode_num = None

        self._reset_log_info()

    def get(self, attr):
        return self.training_env.env_method("get_wrapper_attr", attr)[0]

    def _create_episode_details_dict(self):
        episode_details = {
            "timestep": np.arange(
                self.num_timesteps - len(self.observations),
                self.num_timesteps,
            ),
            "obs": self.observations,
            "action": self.actions,
            "job_wait_reward": self.job_wait_reward,
            "util_reward": self.util_reward,
            "invalid_reward": self.invalid_reward,
            "reward": self.rewards,
            "next_obs": self.new_observations,
        }
        return episode_details

    def _reset_log_info(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.new_observations = []
        self.host_metrics = []
        self.vm_metrics = []
        self.job_metrics = []
        self.job_wait_time = []
        self.job_wait_reward = []
        self.util_reward = []
        self.invalid_reward = []
        self.unutilized_active = []
        self.unutilized_all = []

    def _delete_previous_best(self):
        if self.previous_best_episode_num:
            log_file = os.path.join(
                self.log_dir, f"best_model_actions_{self.previous_best_episode_num}.csv"
            )
            os.remove(log_file)

    def _save_timestep_details(self):
        self.observations.append(self.locals["obs_tensor"][0].cpu())
        self.actions.append(self.locals["actions"][0])
        self.rewards.append(self.locals["rewards"][0])
        self.new_observations.append(self.locals["new_obs"][0])
        self.host_metrics.append(self.locals["infos"][0]["host_metrics"])
        self.vm_metrics.append(self.locals["infos"][0]["vm_metrics"])
        self.job_metrics.append(self.locals["infos"][0]["job_metrics"])
        self.job_wait_time.append(self.locals["infos"][0]["job_wait_time"])
        self.job_wait_reward.append(self.locals["infos"][0]["job_wait_reward"])
        self.util_reward.append(self.locals["infos"][0]["util_reward"])
        self.invalid_reward.append(self.locals["infos"][0]["invalid_reward"])
        self.unutilized_active.append(self.locals["infos"][0]["unutilized_active"])
        self.unutilized_all.append(self.locals["infos"][0]["unutilized_all"])

    def _maybe_save_replay_buffer(self):
        if (
            hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
            and self.save_replay_buffer
        ):
            replay_buffer_path = os.path.join(self.log_dir, "best_model_replay_buffer")
            if self.verbose >= 1:
                print((f"Saving replay buffer to" f"{replay_buffer_path}"))
            self.model.save_replay_buffer(replay_buffer_path)

    def _create_csv_paths(self):
        host_metrics_path = os.path.join(self.log_dir, "host_metrics.csv")
        vm_metrics_path = os.path.join(self.log_dir, "vm_metrics.csv")
        job_metrics_path = os.path.join(self.log_dir, "job_metrics.csv")
        job_wait_time_path = os.path.join(self.log_dir, "job_wait_time.csv")
        unutilized_active_path = os.path.join(self.log_dir, "unutilized_active.csv")
        unutilized_all_path = os.path.join(self.log_dir, "unutilized_all.csv")
        paths = {
            "host_metrics": host_metrics_path,
            "vm_metrics": vm_metrics_path,
            "job_metrics": job_metrics_path,
            "job_wait_time": job_wait_time_path,
            "unutilized_active": unutilized_active_path,
            "unutilized_all": unutilized_all_path,
        }
        return paths

    def _write_simulation_details_to_csvs(self, paths):
        host_metrics_df = pd.DataFrame(self.host_metrics)
        vm_metrics_df = pd.DataFrame(self.vm_metrics)
        job_metrics_df = pd.DataFrame(self.job_metrics)
        job_wait_time_df = pd.DataFrame(self.job_wait_time)
        unutilized_active_df = pd.DataFrame(self.unutilized_active)
        unutilized_all_df = pd.DataFrame(self.unutilized_all)

        host_metrics_df.to_csv(paths["host_metrics"], header=False)
        vm_metrics_df.to_csv(paths["vm_metrics"], header=False)
        job_metrics_df.to_csv(paths["job_metrics"], header=False)
        job_wait_time_df.to_csv(paths["job_wait_time"], header=False)
        unutilized_active_df.to_csv(paths["unutilized_active"], header=False)
        unutilized_all_df.to_csv(paths["unutilized_all"], header=False)

    def _maybe_save_best_episode_details(self):
        if self.save_best_episode_details:
            episode_details = self._create_episode_details_dict()
            df = pd.DataFrame(episode_details)
            episode_details_path = os.path.join(
                self.log_dir,
                f"best_model_actions_{self.get('episode_num') - 1}.csv",
            )
            if self.verbose >= 1:
                print((f"Saving episode details to" f"{episode_details_path}"))
            df.to_csv(episode_details_path, index=False)
            self._delete_previous_best()
            self.previous_best_episode_num = self.get("episode_num") - 1

            csv_paths = self._create_csv_paths()

            if self.verbose >= 1:
                print((f"Saving simulation details to" f"{csv_paths}"))

            self._write_simulation_details_to_csvs(csv_paths)

    def _maybe_print_current_episode_info(self, current_reward):
        if self.verbose >= 1:
            # you can access it also with self.locals['self'].num_timesteps
            # -1 because it is called when reset is already done,
            # so episode number is already incremented
            print(f"Current timesteps: {self.num_timesteps - 1}")
            print(f"Episode number: {self.get('episode_num') - 1}")
            print(f"Episode length: {len(self.observations)}")
            print(f"Best reward: {self.best_reward:.2f}")
            print(f"Current reward: {current_reward:.2f}")

    def _save_new_best(self):
        if self.verbose >= 1:
            print(f"Saving new best model to {self.save_path}")
        self.model.save(self.save_path)
        self._maybe_save_replay_buffer()
        self._maybe_save_best_episode_details()

    def _on_step(self) -> bool:
        # because the environment is a VecEnv environment, the variables are dones
        # and infos instead of done and info. Also, the variables are tuples that
        # allow to return the done and info information for all environments.
        # We know that we have a VecEnv but only 1 environment, so we just take
        # the first element of the tuple.
        self._save_timestep_details()

        if self.locals["dones"][0]:
            print("Env terminated. Printing stat lengths")
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) == 0:
                return True
            # Training reward for this episode
            current_reward = y[-1]
            self._maybe_print_current_episode_info(current_reward)

            # New best model, save the agent and the episode details
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self._save_new_best()

            self._reset_log_info()
        return True
