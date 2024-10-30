import os
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus  # noqa: F401
import torch
import pandas as pd

import stable_baselines3 as sb3
from utils.trace_utils import csv_to_cloudlet_descriptor


def test(hostname, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jobs = csv_to_cloudlet_descriptor(
        os.path.join("mnt", "traces", f"{params["job_trace_filename"]}.csv")
    )

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    best_model_path = os.path.join(
        params["base_log_dir"],
        params["train_model_dir"],
        "best_model",
    )

    # Select the appropriate algorithm
    if hasattr(sb3, params["algorithm"]):
        algorithm = getattr(sb3, params["algorithm"])
    else:
        raise AttributeError(
            f"Algorithm '{params["algorithm"]}' not found in sb3 module."
        )

    # filename_id = generate_filename(params, hostname)

    # Create folder if needed
    os.makedirs(params["log_dir"], exist_ok=True)

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        device=device,
        env=env,
        seed=np.random.randint(0, 2**32 - 1),
    )

    progress_file = os.path.join(params["log_dir"], "progress.csv")

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            params["train_model_dir"],
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)

    episodes_info = {"r": [], "inv": [], "h_a": [], "v_u": [], "j_w": [], "l": []}
    current_step = 0

    obs, info = env.reset()
    print(f"Environment reset. Obs: {obs} Info: {info}")
    episode_reward = 0
    ep_inv_rew = 0
    ep_host_alloc_cores_rew = 0
    ep_vm_unutil_cores_rew = 0
    ep_job_wait_rew = 0
    current_length = 0
    done = False
    while not done:
        current_length += 1
        current_step += 1
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(
        # f"Step: {current_length}, obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}"
        # )
        episode_reward += reward
        ep_inv_rew += info["inv"]
        done = terminated or truncated

    # print(
    # f"Episode ended. Episode length: {current_length}, episode reward: {episode_reward}"
    # )

    # TODO: this may cause OOM because I append the info for every episode
    # and I write it to CSV at the end of all episodes.
    # If I get OOM consider writing one line at a time in the csv, after each episode.
    episodes_info["r"].append(episode_reward)
    episodes_info["l"].append(current_length)

    episode_info_df = pd.DataFrame(episodes_info)
    episode_info_df.to_csv(progress_file, index=False)
    env.close()
