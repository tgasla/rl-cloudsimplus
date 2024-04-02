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

learning_rate_dict = {
    "DQN": "0.00005",
    "DDPG": "0.0001",
    "A2C": "0.0002",
    "PPO": "0.0001"
}

# TODO: I need to pass the trained model filepath as argument and
# then the parameters of the new model (timesteps, env)

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
# parser.add_argument(
#     "model_name_id",
#     type=str,
#     help="The model storage name id"
# )
parser.add_argument(
    "pretraining_environment",
    type=str,
    help="The id of the environment that the agent was trained on"
)
args = parser.parse_args()
algorithm_str = str(args.algorithm).upper()
timesteps = int(args.timesteps)
env_id = str(args.environment)
# model_name_id = str(args.model_name_id)
pretraining_env_id = str(args.pretraining_environment)

filename_id = get_filename_id(
    pretraining_env_id,
    algorithm_str,
    timesteps
)

# Read jobs
swf_reader = SWFReader()
jobs = swf_reader.read("mnt/LLNL-Atlas-2006-2.1-cln.swf", jobs_to_read=25)

model_storage_dir = "./model-storage/"
model_storage_path = model_storage_dir + filename_id

new_filename_id = get_filename_id(
    pretraining_env_id,
    algorithm_str,
    5 * timesteps + timesteps,
    env_id
)

eval_log_dir = "./eval-logs/"
eval_log_path = (
    f"{eval_log_dir}"
    f"{new_filename_id}"
    f"_monitor.csv"
)

tb_log_dir = "./tb-logs/"
tb_log_name = new_filename_id

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
    info_keywords=("cost", "validCount")
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

# Load the trained agent
model = algorithm.load(
    model_storage_path,
    device=device,
    tensorboard_log=tb_log_dir,
    env=env
)

# model.policy.load_state_dict()
model.load_replay_buffer(model_storage_path + "_RP")
# check these fixes
# model._last_obs = None
# model.set_env(env)
# on learn reset_num_timesteps
# save policy
# also check checkpointcallback

model.learning_rate = learning_rate_dict.get(algorithm_str)

# Retrain the agent
model.learn(
    total_timesteps= 5 * timesteps,
    # The right thing to do is to pass True
    # This way, the learning rate resets
    # The only problem is that tensorboard recognizes
    # it as a new model, but that's not a critical issue
    reset_num_timesteps=True,
    tb_log_name=tb_log_name,
    log_interval=1
)

env.close()

new_model_path = (
    f"{model_storage_dir}"
    f"{new_filename_id}"
)

model.save(new_model_path)
model.save_replay_buffer(new_model_path + "_RP")