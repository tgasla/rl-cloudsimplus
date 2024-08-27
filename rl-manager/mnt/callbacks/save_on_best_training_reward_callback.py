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
        save_best_episode_rl_details: bool = True,
        save_best_episode_metrics: bool = True,
        verbose: int = 1,
    ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_replay_buffer = (save_replay_buffer,)
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_reward = -np.inf
        self.save_best_episode_rl_details = save_best_episode_rl_details
        self.save_best_episode_metrics = save_best_episode_metrics
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

    def get(self, attr):
        return self.training_env.env_method("get_wrapper_attr", attr)[0]

    def _on_step(self) -> bool:
        # because the environment is a VecEnv environment, the variables are dones
        # and infos instead of done and info. Also, the variables are tuples that
        # allow to return the done and info information for all environments.
        # We know that we have a VecEnv but only 1 environment, so we just take
        # the first element of the tuple.
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

        print(self.locals)

        if self.locals["dones"][0]:
            print("The env terminated")
            print("Printing stat lengths")
            print(f"host metrics {len(self.host_metrics)}")
            print(f"vm metrics {len(self.vm_metrics)}")
            print(f"job metrics {len(self.job_metrics)}")
            print(f"job wait time {len(self.host_metrics)}")
            print(f"job wait reward {len(self.job_wait_reward)}")
            print(f"util reward {len(self.util_reward)}")
            print(f"invalid reward {len(self.invalid_reward)}")
            print(f"unutilized active {len(self.unutilized_active)}")
            print(f"unutilized all {len(self.unutilized_all)}")
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Training reward for this episode
                current_reward = y[-1]
                if self.verbose >= 1:
                    # you can access it also with self.locals['self'].num_timesteps
                    print(f"Num timesteps: {self.num_timesteps}")
                    # episode_num -1 because it is called when reset is already done, so episode cumber is already incremented
                    print(f"Episode number: {self.get('episode_num') - 1}")
                    # print(f"Episode length: {len(self.observations)}")
                    print(f"Episode length: {len(self.observations)}")
                    print(
                        (
                            f"Best reward: {self.best_reward:.2f} "
                            f"- Current reward: {current_reward:.2f}"
                        )
                    )

                # New best model, you could save the agent here
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    print(f"There are {len(self.observations)} states in this episode")
                    print(f"There are {len(self.actions)} actions in this episode")
                    print(f"There are {len(self.rewards)} rewards in this episode")
                    print(
                        f"There are {len(self.new_observations)} new observations in this episode"
                    )
                    if (
                        hasattr(self.model, "replay_buffer")
                        and self.model.replay_buffer is not None
                        and self.save_replay_buffer
                    ):
                        # If model has a replay buffer, save it
                        replay_buffer_path = os.path.join(
                            self.log_dir, "best_model_replay_buffer"
                        )
                        if self.verbose >= 1:
                            print(self.unutilized_active)
                            print((f"Saving replay buffer to" f"{replay_buffer_path}"))
                        self.model.save_replay_buffer(replay_buffer_path)
                    if self.save_best_episode_rl_details:
                        episode_details = {
                            "obs": self.observations,
                            "action": self.actions,
                            "reward": self.rewards,
                            "next_obs": self.new_observations,
                        }

                        df = pd.DataFrame(episode_details)
                        episode_details_path = os.path.join(
                            self.log_dir, "best_model_actions.csv"
                        )
                        if self.verbose >= 1:
                            print(
                                (f"Saving episode details to" f"{episode_details_path}")
                            )
                        df.to_csv(episode_details_path, index=False)
                    if self.save_best_episode_metrics:
                        host_metrics_str = "host_metrics"
                        vm_metrics_str = "vm_metrics"
                        job_metrics_str = "job_metrics"
                        job_wait_time_str = "job_wait_time"
                        unutilized_active_str = "unutilized_active"
                        unutilized_all_str = "unutilized_all"

                        host_metrics_path = os.path.join(
                            self.log_dir, f"{host_metrics_str}.csv"
                        )
                        vm_metrics_path = os.path.join(
                            self.log_dir, f"{vm_metrics_str}.csv"
                        )
                        job_metrics_path = os.path.join(
                            self.log_dir, f"{job_metrics_str}.csv"
                        )
                        job_wait_time_path = os.path.join(
                            self.log_dir, f"{job_wait_time_str}.csv"
                        )
                        unutilized_active_path = os.path.join(
                            self.log_dir, f"{unutilized_active_str}.csv"
                        )
                        unutilized_all_path = os.path.join(
                            self.log_dir, f"{unutilized_all_str}.csv"
                        )

                        host_metrics_df = pd.DataFrame(self.host_metrics)
                        vm_metrics_df = pd.DataFrame(self.vm_metrics)
                        job_metrics_df = pd.DataFrame(self.job_metrics)
                        # job_wait_time_flattened = [item for sublist in job_wait_time for item in sublist]
                        job_wait_time_df = pd.DataFrame(self.job_wait_time)
                        unutilized_active_df = pd.DataFrame(self.unutilized_active)
                        unutilized_all_df = pd.DataFrame(self.unutilized_all)

                        if self.verbose >= 1:
                            print(
                                (
                                    f"Saving simulation metrics to"
                                    f"{host_metrics_path}, "
                                    f"{vm_metrics_path}, "
                                    f"{job_metrics_path},"
                                    f"{job_wait_time_path},"
                                    f"{unutilized_active_path},"
                                    f"{unutilized_all_path}"
                                )
                            )
                        host_metrics_df.to_csv(host_metrics_path, header=False)
                        vm_metrics_df.to_csv(vm_metrics_path, header=False)
                        job_metrics_df.to_csv(job_metrics_path, header=False)
                        job_wait_time_df.to_csv(job_wait_time_path, header=False)
                        unutilized_active_df.to_csv(
                            unutilized_active_path, header=False
                        )
                        unutilized_all_df.to_csv(unutilized_all_path, header=False)

            self.observations = []
            self.actions = []
            self.rewards = []
            self.new_observations = []
            self.host_metrics = []
            self.vm_metrics = []
            self.job_metrics = []
            self.job_wait_time = []
            self.unutilized_active = []
            self.unutilized_all = []
            self.job_wait_reward = []
            self.util_reward = []
            self.invalid_reward = []
        return True
