import os
import argparse
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from read_swf import SWFReader
import dummy_agents
from utils import get_filename_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "environment", 
    type=str,
    choices=["SmallDC-v0", "LargeDC-v0"],
    help="The id of the environment to train the agent on"
)
parser.add_argument(
    "algorithm", 
    type=str,
    choices=[
        "DQN", "A2C", "PPO", 
        "RNG", "DDPG", "HER", 
        "SAC", "TD3"
    ],
    help="The RL algorithm to train"
)
parser.add_argument(
    "timesteps",
    type=int,
    help="The number of timesteps to train"
)
parser.add_argument(
    "model_name_id",
    type=str,
    help="The model storage name id"
)
parser.add_argument(
    "pretraining_environment",
    type=str,
    help="The id of the environment that the agent was trained on"
)
args = parser.parse_args()
algorithm_str = str(args.algorithm).upper()
timesteps = int(args.timesteps)
env_id = str(parser.environment)
model_name_id = str(parser.model_name_id)
pretraining_env_id = str(parser.pretraining_environment)

# Read jobs
swf_reader = SWFReader()
jobs = swf_reader.read("./LLNL-Atlas-2006-2.1-cln.swf", jobs_to_read=100)

model_storage_dir = f"./model-storage/{pretraining_env_id}/"
model_storage_path = model_storage_dir + model_name_id

eval_log_dir = f"./eval-logs/{pretraining_env_id}/"
eval_log_path = (
    f"{eval_log_dir}"
    f"{model_name_id}"
    f"_monitor.csv"
)

tb_log_dir = f"./tb-logs/{pretraining_env_id}/"
tb_log_name = f"{model_name_id}_{env_id}"

# Create and wrap the environment
env = gym.make(
    env_id,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup="10000",
    split_large_jobs="true",
    render_mode="ansi"
)

# Monitor needs the environment to have a render_mode set
# If render_mode is None, it will give a warning.
env = Monitor(
    env, 
    eval_log_path, 
    override_existing=False
)

# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)
)

if algorithm_str == "RNG":
    algorithm = getattr(dummy_agents, algorithm_str)
    policy = "RngPolicy"
else:
    algorithm = getattr(sb3, algorithm_str)
    policy = "MlpPolicy"

# Instantiate the agent
model = algorithm(
    policy=policy, # TODO: check if we can pass None as a policy
    env=env,
    verbose=True,
    tensorboard_log=tb_log_dir,
    # TODO: for now, all action noise is ignored in RNG algorithm
    action_noise=action_noise,
    device=device
)

# Load the trained agent
model = algorithm.load(model_storage_path, device=device)

# Retrain the agent
model.learn(
    total_timesteps=timesteps,
    progress_bar=False,
    reset_num_timesteps=False,
    tb_log_name=tb_log_name,
    device=device
)

env.close()

new_model_path = (
    f"{model_storage_path}"
    f"_{env_id}"
)

model.save(new_model_path)

os.rename(
    eval_log_path,
    f"{eval_log_dir}"
    f"{model_name_id}"
    f"_{env_id}"
)