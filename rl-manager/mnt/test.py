import os
from datetime import datetime
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch

import stable_baselines3 as sb3
from utils.trace_utils import csv_to_cloudlet_descriptor


def datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algorithm_str = "PPO"
    timesteps = 1000000
    job_trace_filename = "gradual"
    best_model_dir = None  # CHANGE THIS

    jobs = csv_to_cloudlet_descriptor(f"mnt/traces/{job_trace_filename}.csv")

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", jobs_as_json=json.dumps(jobs))

    best_model_path = os.path.join(
        "logs",
        f"{best_model_dir}",
        "best_model",
    )

    # Select the appropriate algorithm
    if hasattr(sb3, algorithm_str):
        algorithm = getattr(sb3, algorithm_str)
    else:
        raise AttributeError(f"Algorithm '{algorithm_str}' not found in sb3 module.")

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        device=device,
        # env=env,
        seed=np.random.randint(0, 2**32 - 1),
    )

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            f"{best_model_dir}",
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)

    for _ in range(timesteps):
        obs, info = env.reset()
        episode_reward = 0
        print(f"Environment reset. Obs: {obs} Info: {info}")
        done = False
        step = 0
        while not done:
            step += 1
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step: {step}, obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}"
            )
            episode_reward += reward
            done = terminated or truncated

        print(
            f"Episode ended. Episode length: {step}, episode reward: {episode_reward}"
        )

    env.close()


if __name__ == "__main__":
    main()
