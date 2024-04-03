import os
from datetime import datetime
import argparse
import json
import numpy as np
import gymnasium as gym
import gym_cloudsimplus
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import dummy_agents
from utils import FilenameFormatter, SWFReader

def datetime_to_str():
    return datetime.now().strftime("%y%m%d-%H%M%S")

learning_rate_dict = {
    "DQN": "0.00005",
    "DDPG": "0.0001",
    "A2C": "0.0002",
    "PPO": "0.0001"
}

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
    "--reward-job-wait-coef",
    type=str,
    help=("The coefficient of the reward function term that is responsible ",
        "for the job waiting penalty"
    )
)
parser.add_argument(
    "--reward-vm-cost-coef",
    type=str,
    help=("The coefficient of the reward function term that is responsible ",
        "for the vm cost penalty"
    )
)
parser.add_argument(
    "--reward-invalid-coef",
    type=str,
    help=("The coefficient of the reward function term that is responsible ",
        "for the invalid action penalty"
    )
)
args = parser.parse_args()
algorithm_str = str(args.algo).upper()
pretrain_env = str(args.pretrain_env)
pretrain_timesteps = int(args.pretrain_timesteps)
transfer_env = str(args.transfer_env)
transfer_timesteps = int(args.transfer_timesteps)
reward_job_wait_coef=str(args.reward_job_wait_coef)
reward_vm_cost_coef=str(args.reward_vm_cost_coef)
reward_invalid_coef=str(args.reward_invalid_coef)

experiment_id = FilenameFormatter.create_filename_id(
    algorithm_str,
    pretrain_env,
    pretrain_timesteps
)

timestamp = datetime_to_str()

filename_id = timestamp + "_" + experiment_id

# Read jobs
jobs = SWFReader.swf_read("mnt/LLNL-Atlas-2006-2.1-cln.swf", jobs_to_read=25)

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
    pretrain_env,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup="1000",
    reward_job_wait_coef=reward_job_wait_coef,
    reward_vm_cost_coef=reward_vm_cost_coef,
    reward_invalid_coef=reward_invalid_coef,
    split_large_jobs="true",
    render_mode="ansi"
)

# Monitor needs the environment to have a render_mode set
# If render_mode is None, it will give a warning.
env = Monitor(
    env, 
    eval_log_path, 
    info_keywords=("validCount", "meanJobWaitPenalty", "meanCostPenalty")
)

# Select the appropriate algorithm
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
    device=device
)

# TODO: A2C and PPO take a n_steps parameter also :) check it out

# Add some action noise for exploration if applicable
if algorithm_str == "DDPG" or \
    algorithm_str == "TD3" or \
    algorithm_str == "SAC":

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.1 * np.ones(n_actions)
    )
    model.action_noise = action_noise

# Train the agent
model.learn(
    total_timesteps=pretrain_timesteps,
    tb_log_name=filename_id,
    log_interval=1
)

# Save the agent
model.save(model_storage_path)

# Save the replay buffer if the algorithm has one
if issubclass(algorithm, OffPolicyAlgorithm):
    model.save_replay_buffer(model_storage_path + "_RP")

# Close the environment and free the resources
env.close()

# Delete model
del model

new_experiment_id = FilenameFormatter.create_filename_id(
    algorithm_str,
    pretrain_env,
    pretrain_timesteps,
    transfer_env,
    transfer_timesteps
)

new_filename_id = timestamp + "_" + new_experiment_id

new_eval_log_path = (
    f"{eval_log_dir}"
    f"{new_filename_id}"
    f"_monitor.csv"
)

new_tb_log_name = new_filename_id

# Create and wrap the environment
new_env = gym.make(
    transfer_env,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup="1000",
    reward_job_wait_coef=reward_job_wait_coef,
    reward_vm_cost_coef=reward_vm_cost_coef,
    reward_invalid_coef=reward_invalid_coef,
    split_large_jobs="true",
    render_mode="ansi"
)

# Monitor needs the environment to have a render_mode set
# If render_mode is None, it will give a warning.
new_env = Monitor(
    new_env, 
    new_eval_log_path, 
    info_keywords=("validCount", "meanJobWaitPenalty", "meanCostPenalty")
)

# Load the trained agent
model = algorithm.load(
    model_storage_path,
    device=device,
    tensorboard_log=tb_log_dir,
    env=new_env
)

# Load the replay buffer if the algorithm has one
if issubclass(algorithm, OffPolicyAlgorithm): 
   model.load_replay_buffer(model_storage_path + "_RP")

# Set the learning rate to a small initial value
model.learning_rate = learning_rate_dict.get(algorithm_str)

# Retrain the agent initializing the weights from the saved agent
model.learn(
    total_timesteps= transfer_timesteps,
    # The right thing to do is to set reset_num_timesteps=True
    # This way, the learning restarts
    # The only problem is that tensorboard recognizes
    # it as a new model, but that's not a critical issue for now
    reset_num_timesteps=True,
    tb_log_name=new_tb_log_name,
    log_interval=1
)

env.close()

# Save the new model
new_model_path = (
    f"{model_storage_dir}"
    f"{new_filename_id}"
)

model.save(new_model_path)

# Save the new replay buffer if the algorithm has one
if issubclass(algorithm, OffPolicyAlgorithm): 
    model.save_replay_buffer(new_model_path + "_RP")