import re
import os
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)
from utils.rl_algorithm_support_flags import (
    ALGORITHMS_WITH_ENT_COEF,
    ALGORITHMS_WITH_ACTION_NOISE,
    ALGORITHMS_WITH_N_STEPS,
)


def get_host_count_from_train_dir(train_model_dir) -> int | None:
    # Regular expression to match the number before "hosts"
    match = re.search(r"(\d+)hosts", train_model_dir)
    if match:
        number = match.group(1)  # Extract the matched number
        return int(number)
    return None


def datacenter_constructor(_, node) -> dict:
    return node.value


def create_logger(save_experiment, log_dir) -> sb3.common.logger.Logger:
    log_destination = ["stdout"]
    if save_experiment:
        log_destination.extend(["csv", "tensorboard"])

    # the logger can write to stdout, progress.csv and tensorboard
    return configure(log_dir, log_destination)


def create_callback(
    save_experiment, log_dir
) -> SaveOnBestTrainingRewardCallback | None:
    if save_experiment:
        return SaveOnBestTrainingRewardCallback(log_dir)
    # the callback writes all the .csv files and saves the model (with replay buffer) when the reward is the best
    return None


def compute_indices(params, prev_host_count=None) -> dict:
    """
    Compute the indices and relevant parameters for freezing weights.

    Args:
        params (dict): Parameters containing host and VM configuration.
        prev_host_count (int, optional): Previous host count for transfer learning. Defaults to None.

    Returns:
        dict: A dictionary containing the computed indices and other derived values.
    """
    max_hosts = params["max_hosts"]
    host_count = params["host_count"]
    host_pes = params["host_pes"]
    small_vm_pes = params["small_vm_pes"]
    min_job_pes = 1

    cur_max_vms = host_count * host_pes // small_vm_pes
    cur_max_jobs = host_count * host_pes // min_job_pes
    max_vms = max_hosts * host_pes // small_vm_pes
    max_jobs = max_hosts * host_pes // min_job_pes
    infr_obs_length = 1 + max_hosts + max_vms + max_jobs
    max_pes_per_node = max_hosts * host_pes

    cur_start_idx = (1 + host_count + cur_max_vms + cur_max_jobs) * (
        max_pes_per_node + 1
    )
    end_idx = infr_obs_length * (max_pes_per_node + 1)

    prev_start_idx = None
    if prev_host_count is not None:
        prev_max_vms = prev_host_count * host_pes // small_vm_pes
        prev_max_jobs = prev_host_count * host_pes // min_job_pes
        prev_start_idx = (1 + prev_host_count + prev_max_vms + prev_max_jobs) * (
            max_pes_per_node + 1
        )

    return {
        "cur_start_idx": cur_start_idx,
        "end_idx": end_idx,
        "prev_start_idx": prev_start_idx,
    }


def maybe_freeze_weights(model, params, prev_host_count=None) -> None:
    """
    Freeze or unfreeze input layer weights based on active/inactive regions.

    Args:
        model (nn.Module): The model whose weights need to be adjusted.
        params (dict): Parameters containing host and VM configuration.
        prev_host_count (int, optional): Previous host count for transfer learning. Defaults to None.
    """
    indices = compute_indices(params, prev_host_count)
    prev_start_idx = indices["prev_start_idx"]
    cur_start_idx = indices["cur_start_idx"]
    end_idx = indices["end_idx"]

    # Access the weights of the input layer
    weights = model.policy.mlp_extractor.policy_net[0].weight

    with torch.no_grad():
        weights[:, cur_start_idx:end_idx].requires_grad = False

        if prev_start_idx is not None:
            weights[:, :cur_start_idx].requires_grad = True
            min_start_idx = min(cur_start_idx, prev_start_idx)
            weights[:, min_start_idx:end_idx] = 0
        else:
            weights[:, cur_start_idx:end_idx] = 0


def vectorize_env(env, algorithm) -> DummyVecEnv | SubprocVecEnv:
    # see https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html note
    if algorithm == "A2C":
        return SubprocVecEnv([lambda: env], start_method="fork")
    return DummyVecEnv([lambda: env])


def get_suitable_device(algorithm) -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() and algorithm != "A2C" else "cpu"
    )


def get_algorithm(
    algorithm_name, vm_allocation_policy
) -> sb3.common.base_class.BaseAlgorithm:
    if vm_allocation_policy == "fromfile" or vm_allocation_policy == "rule-based":
        # If the vm_allocation_policy is fromfile or rule-based, pick a default algorithm
        # so the code triggers the simulation environment creation
        # NOTE: the algorithm decision through learning is not used at all in this case
        algorithm = getattr(sb3, "PPO")
    elif vm_allocation_policy == "rl" or vm_allocation_policy == "bestfit":
        if hasattr(sb3, algorithm_name):
            algorithm = getattr(sb3, algorithm_name)
        elif hasattr(sb3_contrib, algorithm_name):
            algorithm = getattr(sb3_contrib, algorithm_name)
        else:
            raise AttributeError(f"Algorithm {algorithm_name} not found.")
    return algorithm


def maybe_load_replay_buffer(model, train_model_dir) -> None:
    # Load the replay buffer if the algorithm has one
    if hasattr(model, "replay_buffer"):
        best_replay_buffer_path = os.path.join(
            "logs",
            train_model_dir,
            "best_model_replay_buffer",
        )
        model.load_replay_buffer(best_replay_buffer_path)


def create_kwargs_with_algorithm_params(env, params) -> dict:
    algorithm_kwargs = {}
    if params.get("ent_coef") and params["algorithm"] in ALGORITHMS_WITH_ENT_COEF:
        algorithm_kwargs["ent_coef"] = params["ent_coef"]
    if params.get("learning_rate") and params["algorithm"] != "HER":
        algorithm_kwargs["learning_rate"] = params["learning_rate"]
    if params.get("n_rollout_steps") and params["algorithm"] in ALGORITHMS_WITH_N_STEPS:
        algorithm_kwargs["n_steps"] = params["n_rollout_steps"]
    if params.get("seed") and params["algorithm"] != "HER":
        algorithm_kwargs["seed"] = params["seed"]
    if (
        params.get("action_noise")
        and params["algorithm"] in ALGORITHMS_WITH_ACTION_NOISE
    ):
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=params["action_noise"] * np.ones(n_actions)
        )
        algorithm_kwargs["action_noise"] = action_noise
    return algorithm_kwargs


def create_policy_from_obs_space_type(observation_space) -> str:
    if isinstance(observation_space, gym.spaces.Dict):
        return "MultiInputPolicy"  # when state is Spaces.Dict()
    return "MlpPolicy"  # when state is not Spaces.Dict()
