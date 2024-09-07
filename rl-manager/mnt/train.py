import os
from datetime import datetime
import json
import pycurl
from io import BytesIO
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
from utils.parse_config import dict_from_config


def datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    response_buffer = BytesIO()

    hostname = os.getenv("HOSTNAME")

    # Define the socket path and container URL
    unix_socket_path = "/run/docker.sock"
    container_url = f"http://docker/containers/{hostname}/json"

    # Initialize a cURL object
    curl = pycurl.Curl()

    # Set cURL options
    curl.setopt(pycurl.UNIX_SOCKET_PATH, unix_socket_path)
    curl.setopt(pycurl.URL, container_url)
    curl.setopt(pycurl.WRITEFUNCTION, response_buffer.write)
    curl.perform()
    curl.close()

    response_data = response_buffer.getvalue().decode("utf-8")

    replica_id = json.loads(response_data)["Name"].split("-")[-1]

    params = dict_from_config(replica_id, "mnt/config.yml")

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

    experiment_id = generate_filename(
        algorithm_str=algorithm_str,
        pretrain_timesteps=timesteps,
        pretrain_hosts=host_count,
        pretrain_host_pes=host_pes,
        pretrain_host_pe_mips=host_pe_mips,
        pretrain_reward_job_wait_coef=reward_job_wait_coef,
        pretrain_reward_running_vm_cores_coef=reward_running_vm_cores_coef,
        pretrain_reward_unutilized_vm_cores_coef=reward_unutilized_vm_cores_coef,
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
        raise AttributeError(f"Algorithm '{algorithm_str}' not found in sb3 module.")

    timestamp = datetime_to_str()
    filename_id = timestamp + "_" + experiment_id + "_" + hostname
    log_dir = os.path.join(base_log_dir, f"{filename_id}")

    # Read jobs
    jobs = csv_to_cloudlet_descriptor(f"mnt/traces/{job_trace_filename}.csv")
    # print(job_trace_filename, jobs)

    # Create folder if needed
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(
        "SingleDC-v0",
        params=params,
        jobs_as_json=json.dumps(jobs),
    )

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


if __name__ == "__main__":
    main()
