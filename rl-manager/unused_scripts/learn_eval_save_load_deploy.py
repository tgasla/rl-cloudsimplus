import os
import argparse
from datetime import datetime
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch

from read_swf import SWFReader
import dummy_agents
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


def datetime_to_str():
    return datetime.now().strftime("%m%d%y_%H%M%S")


def human_format(num):
    num = float(f"{num:.3f}")
    magnitude = 0
    suffix = ["", "K", "M", "B", "T"]
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    return f"{num:.0f}" + suffix[magnitude]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "environment",
    type=str,
    choices=["SmallDC-v0", "LargeDC-v0"],
    help="The environment to train the agent on",
)
parser.add_argument(
    "algorithm",
    type=str,
    choices=["DQN", "A2C", "PPO", "RNG", "DDPG", "HER", "SAC", "TD3"],
    help="The RL algorithm to train",
)
parser.add_argument("timesteps", type=int, help="The number of timesteps to train")
args = parser.parse_args()
algorithm_str = str(args.algorithm).upper()
timesteps = int(args.timesteps)

# Read jobs
swf_reader = SWFReader()
jobs = swf_reader.read("mnt/LLNL-Atlas-2006-2.1-cln.swf", jobs_to_read=100)

env_id = "LargeDC-v0"

# Create log dir
eval_log_dir = f"./eval-logs/{env_id}/"
os.makedirs(eval_log_dir, exist_ok=True)

eval_log_path = (
    f"{eval_log_dir}"
    f"{algorithm_str}_"
    f"{human_format(timesteps)}_"
    f"{datetime_to_str()}_"
    f"monitor.csv"
)

# Create and wrap the environment
env = gym.make(
    env_id, jobs_as_json=json.dumps(jobs), split_large_jobs="true", render_mode="ansi"
)

# Monitor needs the environment to have a render_mode set
# If render_mode is None, it will give a warning.
env = Monitor(env, eval_log_path, info_keywords=("cost", "validCount", "actionCount"))

# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

if algorithm_str == "RNG":
    algorithm = getattr(dummy_agents, algorithm_str)
    policy = "RngPolicy"
else:
    algorithm = getattr(sb3, algorithm_str)
    policy = "MlpPolicy"

tb_log_dir = f"./tb-logs/{env_id}/"
os.makedirs(eval_log_dir, exist_ok=True)

tb_log_name = f"{algorithm_str}_" f"{human_format(timesteps)}_" f"{datetime_to_str()}_"

# Instantiate the agent
model = algorithm(
    policy=policy,
    env=env,
    verbose=True,
    tensorboard_log=tb_log_dir,
    action_noise=action_noise,
    device=device,
)

# Train the agent
model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=tb_log_name)

# Model evaluation
mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10, render=True
)

print(f"Mean Reward: {mean_reward} +/- {std_reward}")

model_storage_dir = f"./model-storage/{env_id}/"
os.makedirs(model_storage_dir, exist_ok=True)

model_storage_path = (
    f"{model_storage_dir}"
    f"{algorithm_str}_"
    f"{human_format(timesteps)}_"
    f"{datetime_to_str()}"
)

# Save the agent
model.save(model_storage_path)

# Delete trained model to demonstrate loading
del model

# Load the trained agent
model = algorithm.load(model_storage_path)

# Enjoy trained agent
env = gym.make(
    env_id,
    jobs_as_json=json.dumps(jobs),
    split_large_jobs="true",
    render_mode="human",
)
obs, info = env.reset()

cur_timestep = 0
episode_reward = 0

done = False
while not done:
    cur_timestep += 1
    print(f"Current timestep: {cur_timestep}")
    action, _states = model.predict(obs)
    print(f"Action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    episode_reward += reward

    done = terminated or truncated
    print(f"terminated is {terminated}, truncated is {truncated}")
    if terminated:
        print(f"Episode finished! Episode reward: {episode_reward}")
        env.close()
    elif truncated:
        print("Episode truncated. Restarting episode...")
        cur_timestep = 0
        episode_reward = 0
        done = False
        obs, info = env.reset()
