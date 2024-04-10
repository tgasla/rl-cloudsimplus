import os
from datetime import datetime
import argparse
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch
import pandas as pd

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import dummy_agents
from save_on_best_training_reward_callback import SaveOnBestTrainingRewardCallback
from utils import FilenameFormatter, WorkloadUtils

def datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")

learning_rate_dict = {
    "DQN": "0.00005",
    "DDPG": "0.0001",
    "A2C": "0.0002",
    "PPO": "0.0001"
}

monitor_info_keywords = (
    "validCount",
    "meanJobWaitPenalty",
    "meanUtilizationPenalty"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrain-env", 
    type=str,
    choices=["SmallDC-v0", "LargeDC-v0"],
    help="The environment to pretrain the agent on"
)
parser.add_argument(
    "--algo", 
    type=str,
    choices=[
        "DQN", "A2C", "PPO", 
        "RNG", "DDPG", "HER", 
        "SAC", "TD3"
    ],
    help="The RL algorithm that is used for training"
)
parser.add_argument(
    "--pretrain-timesteps",
    type=int,
    help="The number of timesteps to pretrain"
)
parser.add_argument(
    "--transfer-env",
    type=str,
    help="The environment to transfer the agent after pretraining"
)
parser.add_argument(
    "--transfer-timesteps",
    type=int,
    help="The number of timesteps to perform after the environment transfer"
)
parser.add_argument(
    "--simulation-speedup", 
    type=str,
    help="This affects the job arrival time"
)
parser.add_argument(
    "--reward-job-wait-coef",
    type=str,
    help=("The coefficient of the reward function term that is responsible ",
        "for the job waiting penalty"
    )
)
parser.add_argument(
    "--reward-utilization-coef",
    type=str,
    help=("The coefficient of the reward function term that is responsible ",
        "for the utilization penalty"
    )
)
parser.add_argument(
    "--reward-invalid-coef",
    type=str,
    help=("The coefficient of the reward function term that is responsible ",
        "for the invalid action penalty"
    )
)
parser.add_argument(
    "--max-pes-per-job",
    type=str,
    help=("The maximum amount of CPU cores to allow each job to allocate")
)
parser.add_argument(
    "--job-trace-file",
    type=str,
    help=("The filename of the job trace file that will be used for training")
)

args = parser.parse_args()
algorithm_str = str(args.algo).upper()
pretrain_env = str(args.pretrain_env)
pretrain_timesteps = int(args.pretrain_timesteps)
transfer_env = str(args.transfer_env)
transfer_timesteps = int(args.transfer_timesteps)
simulation_speedup = str(args.simulation_speedup)
reward_job_wait_coef=str(args.reward_job_wait_coef)
reward_utilization_coef=str(args.reward_utilization_coef)
reward_invalid_coef=str(args.reward_invalid_coef)
max_pes_per_job=str(args.max_pes_per_job)
job_trace_file=str(args.job_trace_file)

experiment_id = FilenameFormatter.create_filename_id(
    algorithm_str,
    reward_job_wait_coef,
    reward_utilization_coef,
    reward_invalid_coef,
    pretrain_env,
    pretrain_timesteps
)

timestamp = datetime_to_str()

filename_id = timestamp + "_" + experiment_id

# Read jobs
jobs = WorkloadUtils.read_csv(f"mnt/traces/{job_trace_file}.csv")

base_log_dir = "./logs/"

# Create folder if needed
log_dir = f"{base_log_dir}{filename_id}_1"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make(
    pretrain_env,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup=simulation_speedup,
    reward_job_wait_coef=reward_job_wait_coef,
    reward_utilization_coef=reward_utilization_coef,
    reward_invalid_coef=reward_invalid_coef,
    split_large_jobs="true",
    max_pes_per_job=max_pes_per_job,
    job_log_dir=log_dir,
    render_mode="ansi"
)

# Monitor needs the environment to have a render_mode set
# If render_mode is None, it will give a warning.
env = Monitor(
    env, 
    log_dir,
    info_keywords=monitor_info_keywords
)

# Select the appropriate algorithm
if hasattr(sb3, algorithm_str):
    algorithm = getattr(sb3, algorithm_str)
    policy = "MlpPolicy"
else:
    algorithm = getattr(dummy_agents, algorithm_str)
    policy = "RngPolicy"

# Instantiate the agent
model = algorithm(
    policy=policy,
    env=env,
    verbose=True,
    tensorboard_log=base_log_dir,
    device=device
)

# TODO: A2C and PPO take a n_steps parameter also :) check it out

# Add some action noise for exploration if applicable
if hasattr(model, "action_noise"):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)
    )
    model.action_noise = action_noise

callback = SaveOnBestTrainingRewardCallback(
    check_freq=10_000,
    log_dir=log_dir,
    save_replay_buffer=True
)

# Train the agent
model.learn(
    total_timesteps=pretrain_timesteps,
    tb_log_name=filename_id,
    log_interval=1,
    callback=callback
)

# Close the environment and free the resources
env.close()

# Delete model
del model

new_experiment_id = FilenameFormatter.create_filename_id(
    algorithm_str,
    reward_job_wait_coef,
    reward_utilization_coef,
    reward_invalid_coef,
    pretrain_env,
    pretrain_timesteps,
    transfer_env,
    transfer_timesteps
)

new_filename_id = timestamp + "_" + new_experiment_id

# Create folder if needed
new_log_dir = f"{base_log_dir}{new_filename_id}_1"
os.makedirs(new_log_dir, exist_ok=True)

# Create and wrap the environment
new_env = gym.make(
    transfer_env,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup=simulation_speedup,
    reward_job_wait_coef=reward_job_wait_coef,
    reward_utilization_coef=reward_utilization_coef,
    reward_invalid_coef=reward_invalid_coef,
    split_large_jobs="true",
    max_pes_per_job=max_pes_per_job,
    job_log_dir=new_log_dir,
    render_mode="ansi"
)

# Monitor needs the environment to have a render_mode set
# If render_mode is None, it will give a warning.
new_env = Monitor(
    new_env, 
    new_log_dir,
    info_keywords=monitor_info_keywords
)

# Load the trained agent
model = algorithm.load(
    f"{log_dir}/best_model",
    device=device,
    tensorboard_log=base_log_dir,
    env=new_env
)

# Load the replay buffer if the algorithm has one
if hasattr(model, "replay_buffer"):
   model.load_replay_buffer(f"{log_dir}/best_model_replay_buffer")

# Set the learning rate to a small initial value
model.learning_rate = learning_rate_dict.get(algorithm_str)

callback = SaveOnBestTrainingRewardCallback(
    check_freq=10_000,
    log_dir=new_log_dir,
    save_replay_buffer=True
)

# Retrain the agent initializing the weights from the saved agent
model.learn(
    total_timesteps= transfer_timesteps,
    # The right thing to do is to set reset_num_timesteps=True
    # This way, the learning restarts
    # The only problem is that tensorboard recognizes
    # it as a new model, but that's not a critical issue for now
    reset_num_timesteps=True,
    tb_log_name=new_filename_id,
    log_interval=1,
    callback=callback
)

env.close()