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
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)


def transfer(params):
    learning_rate_dict = {
        "DQN": "0.00005",
        "DDPG": "0.0001",
        "A2C": "0.0002",
        "PPO": "0.0001",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jobs = csv_to_cloudlet_descriptor(
        os.path.join("mnt", "traces", f"{params["job_trace_filename"]}.csv")
    )

    # filename_id = generate_filename(params, hostname)

    best_model_path = os.path.join(
        params["base_log_dir"],
        f"{params["train_model_dir"]}",
        "best_model",
    )

    # Select the appropriate algorithm
    if hasattr(sb3, params["algorithm"]):
        algorithm = getattr(sb3, params["algorithm"])
    else:
        raise AttributeError(
            f"Algorithm '{params["algorithm"]}' not found in sb3 module."
        )

    # log_dir = os.path.join(base_log_dir, f"{filename_id}")

    # Create folder if needed
    os.makedirs(params["log_dir"], exist_ok=True)

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    menv = Monitor(env, params["log_dir"])

    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if params["algorithm"] == "A2C":
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

    logger = configure(params["log_dir"], ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            f"{params["train_model_dir"]}",
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)

    # Set the learning rate to a small initial value
    model.learning_rate = learning_rate_dict.get(params["algorithm"])

    callback = SaveOnBestTrainingRewardCallback(
        log_dir=params["log_dir"],
    )

    # Retrain the agent initializing the weights from the saved agent
    model.learn(
        total_timesteps=int(params["timesteps"]),
        # The right thing to do is to set reset_num_timesteps=True
        # This way, the learning restarts
        # The only problem is that tensorboard recognizes
        # it as a new model, but that's not a critical issue for now
        # see: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
        reset_num_timesteps=True,
        log_interval=1,
        callback=callback,
    )

    env.close()
    del model
