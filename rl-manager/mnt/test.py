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

    # if pretrain dir is blank then you should pretrain first,
    # otherwise load agent from pretrain dir

    jobs = csv_to_cloudlet_descriptor("mnt/traces/50_50jobs_2bursts.csv")

    # Create and wrap the environment
    env = gym.make(
        "SingleDC-v0",
        max_timesteps_per_episode="100",
        datacenter_hosts_cnt="10",
        host_pe_mips="10",
        host_pes="10",
        reward_job_wait_coef="0.3",
        reward_util_coef="0.3",
        reward_invalid_coef="0.4",
        jobs_as_json=json.dumps(jobs),
        max_job_pes="1",
        render_mode="ansi",
    )

    best_model_path = os.path.join(
        "logs/240825-172141_PPO_8M_10H_10P_10M_100jobs_fast_1MJC_0.3Q_0.3U_0.4I",
        "best_model",
    )

    # Load the trained agent
    model = sb3.PPO.load(
        best_model_path,
        device=device,
        # env=env,
        seed=np.random.randint(0, 2**32 - 1),
    )

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs/240825-172141_PPO_8M_10H_10P_10M_100jobs_fast_1MJC_0.3Q_0.3U_0.4I",
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)

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

    print(f"Episode ended. Episode length: {step}, episode reward: {episode_reward}")
    env.close()


if __name__ == "__main__":
    main()
