import os
import numpy as np
import pandas as pd
# from stable_baselines3.common import results_plotter
#from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
	"""
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq:
	:param log_dir: Path to the folder where the model will be saved.
		It must contains the file created by the ``Monitor`` wrapper.
	:param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
	"""
	def __init__(
		self,
		check_freq: int,
		log_dir: str,
		save_replay_buffer: bool = True,
		save_best_episode_rl_details: bool = True,
		save_best_episode_metrics: bool = True,
		verbose: int = 1
	):
		super().__init__(verbose)
		self.check_freq = check_freq
		self.log_dir = log_dir
		self.save_replay_buffer = save_replay_buffer,
		self.save_path = os.path.join(log_dir, "best_model")
		self.best_reward = -np.inf
		self.save_best_episode_rl_details = save_best_episode_rl_details
		self.save_best_episode_metrics = save_best_episode_metrics

	def get(self, attr):
		return self.training_env.env_method("get_wrapper_attr", attr)[0]

	def _on_step(self) -> bool:
		if self.n_calls % self.check_freq == 0:

			# Retrieve training reward
			x, y = ts2xy(load_results(self.log_dir), "timesteps")
			if len(x) > 0:
				# Training reward for this episode
				reward = y[-1]
				# print(last_reward)
				if self.verbose >= 1:
					print(f"Num timesteps: {self.num_timesteps}")
					print((
						f"Best reward: {self.best_reward:.2f} "
						f"- Last reward: {reward:.2f}"
					))

				# New best model, you could save the agent here
				if reward > self.best_reward:
					self.best_reward = reward
					# Example for saving best model
					if self.verbose >= 1:
						print(f"Saving new best model to {self.save_path}")
						self.model.save(self.save_path)
					if hasattr(self.model, "replay_buffer") \
						and self.model.replay_buffer is not None \
						and self.save_replay_buffer:
						# If model has a replay buffer, save it
						replay_buffer_path = os.path.join(
							self.log_dir,
							"best_model_replay_buffer"
						)
						if self.verbose >= 1:
							print((
								f"Saving replay buffer to"
		 						f"{replay_buffer_path}"
							))
						self.model.save_replay_buffer(replay_buffer_path)
					if self.save_best_episode_rl_details:
						episode_details = self.get("episode_details")
						del episode_details["state"][-1]

						df = pd.DataFrame(episode_details)
						episode_details_path = os.path.join(
							self.log_dir,
							"best_model_actions.csv"
						)
						if self.verbose >= 1:
							print((
								f"Saving episode details to"
								f"{episode_details_path}"
							))
						df.to_csv(episode_details_path)
					if self.save_best_episode_metrics:
						host_metrics_str = "host_metrics"
						vm_metrics_str = "vm_metrics"
						job_metrics_str = "job_metrics"
						job_wait_time_str = "job_wait_time"
						unutilized_active_str = "unutilized_active"
						unutilized_all_str = "unutilized_all"

						host_metrics_path = os.path.join(self.log_dir, f"{host_metrics_str}.csv")
						vm_metrics_path = os.path.join(self.log_dir, f"{vm_metrics_str}.csv")
						job_metrics_path = os.path.join(self.log_dir, f"{job_metrics_str}.csv")
						job_wait_time_path = os.path.join(self.log_dir, f"{job_wait_time_str}.csv")
						unutilized_active_path = os.path.join(self.log_dir, f"{unutilized_active_str}.csv")
						unutilized_all_path = os.path.join(self.log_dir, f"{unutilized_all_str}.csv")

						host_metrics = self.get(host_metrics_str)
						host_metrics_df = pd.DataFrame(host_metrics)
						vm_metrics = self.get(vm_metrics_str)
						vm_metrics_df = pd.DataFrame(vm_metrics)
						job_metrics = self.get(job_metrics_str)
						job_metrics_df = pd.DataFrame(job_metrics)
						job_wait_time = self.get(job_wait_time_str)
						job_wait_time_flattened = [item for sublist in job_wait_time for item in sublist]
						job_wait_time_df = pd.DataFrame(job_wait_time_flattened)
						unutilized_active = self.get(unutilized_active_str)
						unutilized_active_df = pd.DataFrame(unutilized_active)
						unutilized_all = self.get(unutilized_all_str)
						unutilized_all_df = pd.DataFrame(unutilized_all)
						
						if self.verbose >= 1:
							print((
								f"Saving simulation metrics to"
								f"{host_metrics_path}, "
								f"{vm_metrics_path}, "
								f"{job_metrics_path},"
								f"{job_wait_time_path},"
								f"{unutilized_active_path},"
								f"{unutilized_all_path}"
							))
						host_metrics_df.to_csv(host_metrics_path)
						vm_metrics_df.to_csv(vm_metrics_path)
						job_metrics_df.to_csv(job_metrics_path)
						job_wait_time_df.to_csv(job_wait_time_path, index=False)
						unutilized_active_df.to_csv(unutilized_active_path, index=False)
						unutilized_all_df.to_csv(unutilized_all_path, index=False)
		return True