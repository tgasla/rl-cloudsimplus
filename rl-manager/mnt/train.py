import os
from datetime import datetime
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import custom_agents
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)

from utils.filename_generator import generate_filename

# from utils.argparser import parse_args
from utils.trace_utils import csv_to_cloudlet_descriptor


def datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def main():
    monitor_info_keywords = (
        "ep_job_wait_rew_mean",
        "ep_util_rew_mean",
        "ep_valid_count",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args = parse_args()
    algorithm_str = os.getenv("ALGO")
    timesteps = os.getenv("TRAINING_TIMESTEPS")
    hosts_count = os.getenv("HOSTS_COUNT")
    host_pes = os.getenv("HOST_PES")
    host_pe_mips = os.getenv("HOST_PE_MIPS")
    reward_job_wait_coef = os.getenv("REWARD_JOB_WAIT_COEF")
    reward_util_coef = os.getenv("REWARD_UTIL_COEF")
    reward_invalid_coef = os.getenv("REWARD_INVALID_COEF")
    max_job_pes = os.getenv("MAX_JOB_PES")
    job_trace_filename = os.getenv("JOB_TRACE_FILENAME")

    experiment_id = generate_filename(
        algorithm_str=algorithm_str,
        pretrain_timesteps=timesteps,
        pretrain_hosts=hosts_count,
        pretrain_host_pes=host_pes,
        pretrain_host_pe_mips=host_pe_mips,
        pretrain_reward_job_wait_coef=reward_job_wait_coef,
        pretrain_reward_util_coef=reward_util_coef,
        pretrain_reward_invalid_coef=reward_invalid_coef,
        pretrain_job_trace_filename=job_trace_filename,
        pretrain_max_job_pes=max_job_pes,
    )

    base_log_dir = "./logs/"

    # Select the appropriate algorithm
    if hasattr(sb3, algorithm_str):
        algorithm = getattr(sb3, algorithm_str)
        policy = "MlpPolicy"
    else:
        algorithm = getattr(custom_agents, algorithm_str)
        policy = "RngPolicy"

    timestamp = datetime_to_str()
    filename_id = timestamp + "_" + experiment_id
    log_dir = os.path.join(base_log_dir, f"{filename_id}")
    # Read jobs
    jobs = csv_to_cloudlet_descriptor(f"mnt/traces/{job_trace_filename}.csv")
    print(job_trace_filename, jobs)

    # Create folder if needed
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", jobs_as_json=json.dumps(jobs))

    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    menv = Monitor(env, log_dir, info_keywords=monitor_info_keywords)

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


if __name__ == "__main__":
    main()
