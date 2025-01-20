from gc import freeze
import os
import json
import gymnasium as gym
import gym_cloudsimplus  # noqa: F401
import torch
import numpy as np
import re

import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

from utils.trace_utils import csv_to_cloudlet_descriptor
from utils.rl_algorithm_support_flags import (
    ALGORITHMS_WITH_ENT_COEF,
    ALGORITHMS_WITH_ACTION_NOISE,
    ALGORITHMS_WITH_N_STEPS,
)
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)


def freeze_inactive_input_layer_weights(model, params):
    max_hosts = params["max_hosts"]
    host_pes = params["host_pes"]
    small_vm_pes = params["small_vm_pes"]
    min_job_pes = 1
    prev_host_count = get_host_count_from_train_dir(params["train_model_dir"])
    prev_max_vms = prev_host_count * host_pes // small_vm_pes
    prev_max_jobs = prev_host_count * host_pes // min_job_pes
    cur_host_count = params["host_count"]
    cur_max_vms = cur_host_count * host_pes // small_vm_pes
    cur_max_jobs = cur_host_count * host_pes // min_job_pes
    max_vms = max_hosts * host_pes // small_vm_pes
    max_jobs = max_hosts * host_pes // min_job_pes
    infr_obs_length = 1 + max_hosts + max_vms + max_jobs
    max_pes_per_node = max_hosts * host_pes

    # Calculate the start and end indices for the last 700 elements of Part 1
    prev_start_idx = (1 + prev_host_count + prev_max_vms + prev_max_jobs) * (
        max_pes_per_node + 1
    )
    cur_start_idx = (1 + cur_host_count + cur_max_vms + cur_max_jobs) * (
        max_pes_per_node + 1
    )
    end_idx = infr_obs_length * (max_pes_per_node + 1)  # Exclusive

    # Access the weights of the input layer
    weights = model.policy.mlp_extractor.policy_net[0].weight

    with torch.no_grad():
        weights[:, cur_start_idx:end_idx].requires_grad = False
        weights[:, :cur_start_idx].requires_grad = True
        min_start_idx = min(cur_start_idx, prev_start_idx)
        weights[:, min_start_idx:end_idx] = 0


def create_logger(save_experiment, log_dir):
    log_destination = ["stdout"]
    if save_experiment:
        os.makedirs(log_dir, exist_ok=True)
        log_destination.extend(["csv", "tensorboard"])
    return configure(log_dir, log_destination)


def create_callback(save_experiment, log_dir):
    if save_experiment:
        return SaveOnBestTrainingRewardCallback(log_dir)
    return None


def get_host_count_from_train_dir(train_model_dir):
    # Regular expression to match the number before "hosts"
    match = re.search(r"(\d+)hosts", train_model_dir)
    if not match:
        match = re.search(r"(\d+)nodes", train_model_dir)
    if not match:
        return None
    number = match.group(1)  # Extract the matched number
    return int(number)


def transfer(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jobs = csv_to_cloudlet_descriptor(
        os.path.join("mnt", "traces", f"{params['job_trace_filename']}")
    )

    best_model_path = os.path.join(
        params["base_log_dir"],
        f"{params['train_model_dir']}",
        "best_model",
    )

    # Select the appropriate algorithm
    if hasattr(sb3, params["algorithm"]):
        algorithm = getattr(sb3, params["algorithm"])
    elif hasattr(sb3_contrib, params["algorithm"]):
        algorithm = getattr(sb3_contrib, params["algorithm"])
    else:
        raise AttributeError(f"Algorithm {params['algorithm']} not found.")

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs_as_json=json.dumps(jobs))

    env = Monitor(env, params["log_dir"])

    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if params["algorithm"] == "A2C":
        device = "cpu"
        env = SubprocVecEnv([lambda: env], start_method="fork")
    else:
        env = DummyVecEnv([lambda: env])

    # Change any model parameters you want here
    custom_objects = {}

    if params.get("learning_rate"):
        custom_objects["learning_rate"] = params["learning_rate"]
    if params.get("ent_coef") and params["algorithm"] in ALGORITHMS_WITH_ENT_COEF:
        custom_objects["ent_coef"] = params["ent_coef"]
    if (
        params.get("action_noise")
        and params["algorithm"] in ALGORITHMS_WITH_ACTION_NOISE
    ):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=params["action_noise"] * np.ones(n_actions)
        )
        custom_objects["action_noise"] = action_noise
    if params.get("n_rollout_steps") and params["algorithm"] in ALGORITHMS_WITH_N_STEPS:
        custom_objects["n_steps"] = params["n_rollout_steps"]
    if params.get("seed") and params["algorithm"] != "HER":
        custom_objects["seed"] = params["seed"]

    # Load the trained agent
    model = algorithm.load(
        best_model_path,
        device=device,
        env=env,
        custom_objects=custom_objects,
    )

    freeze_inactive_input_layer_weights(model, params)

    logger = create_logger(params["save_experiment"], params["log_dir"])
    callback = create_callback(params["save_experiment"], params["log_dir"])

    model.set_logger(logger)

    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            f"{params['train_model_dir']}",
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)

    # Retrain the agent initializing the weights from the saved agent
    # The right thing to do is to set reset_num_timesteps=True
    # This way, the learning restarts
    # The only problem is that tensorboard recognizes
    # it as a new model, but that's not a critical issue for now
    # see: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    model.learn(
        total_timesteps=params["timesteps"],
        reset_num_timesteps=True,
        log_interval=1,
        callback=callback,
    )

    env.close()
    del model
