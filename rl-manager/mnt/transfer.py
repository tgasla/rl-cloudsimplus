import os
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus  # noqa: F401
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from utils.trace_utils import csv_to_cloudlet_descriptor
from utils.filename_generator import generate_filename
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)


def transfer(hostname, params):
    learning_rate_dict = {
        "DQN": "0.00005",
        "DDPG": "0.0001",
        "A2C": "0.0002",
        "PPO": "0.0001",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algorithm_str = params["algorithm"]
    timesteps = params["timesteps"]
    job_trace_filename = params["job_trace_filename"]
    host_count = params["host_count"]
    host_pes = params["host_pes"]
    host_pe_mips = params["host_pe_mips"]
    reward_job_wait_coef = params["reward_job_wait_coef"]
    reward_running_vm_cores_coef = params["reward_running_vm_cores_coef"]
    reward_unutilized_vm_cores_coef = params["reward_unutilized_vm_cores_coef"]
    reward_invalid_coef = params["reward_invalid_coef"]
    max_job_pes = params["max_job_pes"]
    train_model_dir = params["train_model_dir"]

    jobs = csv_to_cloudlet_descriptor(f"mnt/traces/{job_trace_filename}.csv")

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
        train_model_dir=train_model_dir,
        mode="transfer",
        hostname=hostname,
    )

    base_log_dir = "./logs/"

    best_model_path = os.path.join(
        base_log_dir,
        f"{train_model_dir}",
        "best_model",
    )

    # Select the appropriate algorithm
    if hasattr(sb3, algorithm_str):
        algorithm = getattr(sb3, algorithm_str)
    else:
        raise AttributeError(f"Algorithm '{algorithm_str}' not found in sb3 module.")

    log_dir = os.path.join(base_log_dir, f"{filename_id}")

    # Create folder if needed
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    menv = Monitor(env, log_dir)

    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if algorithm_str == "A2C":
        device = "cpu"
        venv = SubprocVecEnv([lambda: menv], start_method="fork")
    else:
        venv = DummyVecEnv([lambda: menv])

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        device=device,
        env=venv,
        seed=np.random.randint(0, 2**32 - 1),
    )

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            f"{train_model_dir}",
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)

    # Set the learning rate to a small initial value
    model.learning_rate = learning_rate_dict.get(algorithm_str)

    callback = SaveOnBestTrainingRewardCallback(
        log_dir=log_dir,
    )

    # Retrain the agent initializing the weights from the saved agent
    model.learn(
        total_timesteps=int(timesteps),
        # The right thing to do is to set reset_num_timesteps=True
        # This way, the learning restarts
        # The only problem is that tensorboard recognizes
        # it as a new model, but that's not a critical issue for now
        reset_num_timesteps=True,
        log_interval=1,
        callback=callback,
    )

    env.close()
    del model
