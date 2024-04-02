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

# TODO: I should not have two independent files for pretarining and retraining
# This way I also avoid saving and reeloading the model as this is 
# not really needed because we do not want (at least for now) to transfer
#the agent to a different machine. We should follow the folloiwng procedure:
# In a single code file, 
#   (1) instantiate the pretraining environment (env.make(..))
#   (2) instantiate and pretrain the agent (model = algo(..) and model.learn)
#   (3) close the environment (env.close())
#   (4) instantiate the retraining environment (env.make(..))
#   (5) set the new environment to the existing model (model.set_env(env))
#   (6) do retraining (model.learn(..))
# ATTENTION: If you really need to save and load the agent, wyou should
# also save and load the replay  buffers for the off-policy algorithms.
# The most clean way to do it is by checking if an algorithm inherits from
# the OffPolicyAlgorithm class and if so, do model.load_replay_buffer(..)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "environment", 
    type=str,
    choices=["SmallDC-v0", "LargeDC-v0"],
    help="The environment to train the agent on"
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
args = parser.parse_args()
algorithm_str = str(args.algorithm).upper()
timesteps = int(args.timesteps)
env_id = str(args.environment)

filename_id = get_filename_id(
    env_id,
    algorithm_str,
    timesteps
)

# Read jobs
swf_reader = SWFReader()
jobs = swf_reader.read("mnt/LLNL-Atlas-2006-2.1-cln.swf", jobs_to_read=25)

# Create eval dir
eval_log_dir = "./eval-logs/"
os.makedirs(eval_log_dir, exist_ok=True)

eval_log_path = (
    f"{eval_log_dir}"
    f"{filename_id}"
    f"_monitor.csv"
)

# Create tensorboard dir
tb_log_dir = "./tb-logs/"
os.makedirs(tb_log_dir, exist_ok=True)

# Create model storage dir
model_storage_dir = "./model-storage/"
os.makedirs(model_storage_dir, exist_ok=True)

model_storage_path = (
    f"{model_storage_dir}"
    f"{filename_id}"
)

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
    info_keywords=("cost","validCount")
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
    policy=policy,
    env=env,
    verbose=True,
    tensorboard_log=tb_log_dir,
    # TODO: for now, all action noise is ignored in RNG algorithm
    action_noise=action_noise,
    device=device
)

# Train the agent
model.learn(
    total_timesteps=timesteps,
    progress_bar=False,
    tb_log_name=filename_id,
    log_interval=1
)

# Save the agent
model.save(model_storage_path)

model.save_replay_buffer(model_storage_path + "_RP")

# Close the environment and free the resources
env.close()
