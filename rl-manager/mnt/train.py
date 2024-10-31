import os
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus  # noqa: F401
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)

from utils.trace_utils import csv_to_cloudlet_descriptor


def train(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(params)
    # filename_id = generate_filename(params)

    # Select the appropriate algorithm
    if (
        params["vm_allocation_policy"] == "fromfile"
        or params["vm_allocation_policy"] == "heuristic"
    ):
        # If the vm_allocation_policy is fromfile or heuristic, pick a default algorithm
        # so the code triggers the simulation environment creation
        # NOTE: the algorithm decision through learning is not used at all in this case
        algorithm = getattr(sb3, "PPO")
        policy = "MlpPolicy"
    elif params["vm_allocation_policy"] == "rl" and hasattr(sb3, params["algorithm"]):
        algorithm = getattr(sb3, params["algorithm"])
        policy = "MultiInputPolicy"
    else:
        raise AttributeError(f"Algorithm {params['algorithm']} not found.")

    # if hasattr(algorithm, "ent_coef"):
    # algorithm.ent_coef = 0.01
    # algorithm.learning_rate=0.1,
    # algorithm.clip_range=0.7,
    # algorithm.n_steps=1024

    # Read jobs
    job_trace_path = os.path.join(
        "mnt", "traces", f"{params['job_trace_filename']}.csv"
    )
    jobs = csv_to_cloudlet_descriptor(job_trace_path)

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    log_destination = ["stdout"]
    callback = None
    if params["save_experiment"]:
        log_destination.extend(["csv", "tensorboard"])
        # the callback writes all the other .csv files and saves the model (with replay buffer) when the reward is the best
        callback = SaveOnBestTrainingRewardCallback(log_dir=params["log_dir"])

    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    #   add info_keywords if needed
    # If log_dir is None, it will not log anything
    env = Monitor(env, params["log_dir"])

    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if params["algorithm"] == "A2C":
        device = "cpu"
        env = SubprocVecEnv([lambda: env], start_method="fork")
    else:
        env = DummyVecEnv([lambda: env])

    # Instantiate the agent
    model = algorithm(policy=policy, device=device, env=env, seed=params["seed"])

    # the logger can write to stdout, progress.csv and tensorboard
    logger = configure(params["log_dir"], log_destination)
    model.set_logger(logger)

    # Add some action noise for exploration if applicable
    if hasattr(model, "action_noise"):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model.action_noise = action_noise

    # Train the agent
    model.learn(total_timesteps=params["timesteps"], log_interval=1, callback=callback)

    # Close the environment and free the resources
    env.close()

    # Delete the model from memory
    del model
