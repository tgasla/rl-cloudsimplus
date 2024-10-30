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
        policy = "MlpPolicy"
    else:
        raise AttributeError(f"Algorithm '{params["algorithm"]}' not found.")

    # Read jobs
    job_trace_path = os.path.join(
        "mnt", "traces", f"{params["job_trace_filename"]}.csv"
    )
    jobs = csv_to_cloudlet_descriptor(job_trace_path)

    # print(job_trace_filename, jobs)

    # Create folder if needed
    if params["log_experiment"]:
        os.makedirs(params["log_dir"], exist_ok=True)

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    # add info_keywords if needed
    if params["log_experiment"]:
        env = Monitor(env, params["log_dir"])

    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if params["algorithm"] == "A2C":
        device = "cpu"
        venv = SubprocVecEnv([lambda: env], start_method="fork")
    else:
        venv = DummyVecEnv([lambda: env])

    # if hasattr(algorithm, "ent_coef"):
    # algorithm.ent_coef = 0.01
    # algorithm.learning_rate=0.1,
    # algorithm.clip_range=0.7,
    # algorithm.n_steps=1024

    # Instantiate the agent
    model = algorithm(
        policy=policy,
        env=venv,
        device=device,
        seed=np.random.randint(0, 2**32 - 1),
    )

    if params["log_experiment"]:
        # the logger can write to stdout, progress.csv and tensorboard
        logger = configure(params["log_dir"], ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

    # Add some action noise for exploration if applicable
    if hasattr(model, "action_noise"):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model.action_noise = action_noise

    callback = None
    if params["log_experiment"]:
        # the callback writes all the other .csv files and saves the model (with replay buffer) when the reward is the best
        callback = SaveOnBestTrainingRewardCallback(log_dir=params["log_dir"])

    # Train the agent
    model.learn(
        total_timesteps=int(params["timesteps"]),
        log_interval=1,
        callback=callback,
    )

    # Close the environment and free the resources
    env.close()

    # Delete model
    del model
