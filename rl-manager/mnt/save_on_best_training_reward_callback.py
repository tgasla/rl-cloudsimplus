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
		save_best_episode_details: bool = True,
		verbose: int = 1
	):    
		super().__init__(verbose)
		self.check_freq = check_freq
		self.log_dir = log_dir
		self.save_replay_buffer = save_replay_buffer,
		self.save_path = os.path.join(log_dir, "best_model")
		self.best_mean_reward = -np.inf
		self.save_best_episode_details = save_best_episode_details
	
	def get(self, attr):
		return self.training_env.env_method("get_wrapper_attr", attr)

	def _on_step(self) -> bool:
		if self.n_calls % self.check_freq == 0:

			# Retrieve training reward
			x, y = ts2xy(load_results(self.log_dir), "timesteps")
			if len(x) > 0:
				# Mean training reward over the last 100 episodes
				mean_reward = np.mean(y[-100:])
				if self.verbose >= 1:
					print(f"Num timesteps: {self.num_timesteps}")
					print((
						f"Best mean reward: {self.best_mean_reward:.2f} "
						f"- Last mean reward per episode: {mean_reward:.2f}"
					))

				# New best model, you could save the agent here
				if mean_reward > self.best_mean_reward:
					self.best_mean_reward = mean_reward
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
					if self.save_best_episode_details:
						episode_details = self.get("episode_details")[0]
						del episode_details["state"][-1]
						
						df = pd.DataFrame(episode_details)
						episode_details_path = os.path.join(
							self.log_dir, 
							"best_model_actions.csv"
						)
						df.to_csv(episode_details_path)
		return True