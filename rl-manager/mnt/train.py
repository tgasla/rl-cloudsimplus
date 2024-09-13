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

from utils.filename_generator import generate_filename
from utils.trace_utils import csv_to_cloudlet_descriptor


def train(hostname, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algorithm_str = params["algorithm"]
    timesteps = params["timesteps"]
    host_count = params["host_count"]
    host_pes = params["host_pes"]
    host_pe_mips = params["host_pe_mips"]
    reward_job_wait_coef = params["reward_job_wait_coef"]
    reward_running_vm_cores_coef = params["reward_running_vm_cores_coef"]
    reward_unutilized_vm_cores_coef = params["reward_unutilized_vm_cores_coef"]
    reward_invalid_coef = params["reward_invalid_coef"]
    max_job_pes = params["max_job_pes"]
    job_trace_filename = params["job_trace_filename"]

    filename_id = generate_filename(
        algorithm_str=algorithm_str,
        timesteps=timesteps,
        hosts=host_count,
        host_pes=host_pes,
        host_pe_mips=host_pe_mips,
        reward_job_wait_coef=reward_job_wait_coef,
        reward_running_vm_cores_coef=reward_running_vm_cores_coef,
        reward_unutilized_vm_cores_coef=reward_unutilized_vm_cores_coef,
        reward_invalid_coef=reward_invalid_coef,
        job_trace_filename=job_trace_filename,
        max_job_pes=max_job_pes,
        mode="train",
        hostname=hostname,
    )

    base_log_dir = "./logs/"

    # Select the appropriate algorithm
    if hasattr(sb3, algorithm_str):
        algorithm = getattr(sb3, algorithm_str)
        policy = "MlpPolicy"
    else:
        raise AttributeError(f"Algorithm '{algorithm_str}' not found in sb3 module.")

    log_dir = os.path.join(base_log_dir, f"{filename_id}")

    # Read jobs
    job_trace_path = os.path.join("mnt", "traces", f"{job_trace_filename}.csv")
    jobs = csv_to_cloudlet_descriptor(job_trace_path)
    # print(job_trace_filename, jobs)

    # Create folder if needed
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    # add info_keywords if needed
    menv = Monitor(env, log_dir)

    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if algorithm_str == "A2C":
        device = "cpu"
        venv = SubprocVecEnv([lambda: menv], start_method="fork")
    else:
        venv = DummyVecEnv([lambda: menv])

    # if hasattr(algorithm, "ent_coef"):
    # algorithm.ent_coef = 0.01
    # algorithm.learning_rate=0.1,
    # algorithm.clip_range=0.7,
    # algorithm.n_steps=1024

    # Instantiate the agent
    model = algorithm(
        policy=policy,
        env=venv,
        verbose=1,
        device=device,
        seed=np.random.randint(0, 2**32 - 1),
    )

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # Add some action noise for exploration if applicable
    if hasattr(model, "action_noise"):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model.action_noise = action_noise

    callback = SaveOnBestTrainingRewardCallback(log_dir=log_dir)

    # Train the agent
    model.learn(
        total_timesteps=int(timesteps),
        log_interval=1,
        callback=callback,
    )

    # Close the environment and free the resources
    env.close()

    # Delete model
    del model
