import re
import os
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from callbacks.save_on_best_training_reward_callback import (
    SaveOnBestTrainingRewardCallback,
)
from utils.rl_algorithm_statics import (
    ALGORITHMS_WITH_ENT_COEF,
    ALGORITHMS_WITH_ACTION_NOISE,
    ALGORITHMS_WITH_N_STEPS,
)


class DictLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Generalized feature extractor for dictionary observations with Box and MultiDiscrete spaces.
    Uses LSTM for sequential processing.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embedding_size: int = 32,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()  # Store embeddings for MultiDiscrete keys
        total_input_dim = 0

        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, spaces.Box):
                # Linear layer for Box space
                input_dim = subspace.shape[0]
                self.extractors[key] = nn.Linear(input_dim, embedding_size)
                total_input_dim += embedding_size

            elif isinstance(subspace, spaces.MultiDiscrete):
                # Create embeddings for each MultiDiscrete key separately
                num_categories = subspace.nvec
                embedding_dims = [
                    min(embedding_size, (n // 2) + 1) for n in num_categories
                ]

                self.embeddings[key] = nn.ModuleList(
                    [
                        nn.Embedding(num_categories[i], embedding_dims[i])
                        for i in range(len(num_categories))
                    ]
                )

                total_input_dim += sum(embedding_dims)

        # LSTM for sequential processing
        self.lstm = nn.LSTM(total_input_dim, 64, batch_first=True)

        # Final FC layer
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        extracted_features = []

        # Process Box features
        for key, extractor in self.extractors.items():
            if key in observations:
                extracted_features.append(extractor(observations[key]))

        # Process MultiDiscrete features
        for key, embedding_layers in self.embeddings.items():
            categorical_features = []
            for i, embedding in enumerate(embedding_layers):
                categorical_features.append(embedding(observations[key].long()[:, i]))
            extracted_features.append(torch.cat(categorical_features, dim=-1))

        # Concatenate all extracted features
        concatenated = torch.cat(extracted_features, dim=-1)

        # Ensure correct LSTM input shape: (batch_size, seq_len, input_dim)
        batch_size = next(iter(observations.values())).shape[
            0
        ]  # Get batch size from any input
        concatenated = concatenated.view(batch_size, 1, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(concatenated)

        # Take last LSTM output and pass through FC layer
        return self.fc(lstm_out[:, -1, :])


class BoxLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Box observations with sequential processing using LSTM.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # The input dimension is the flattened shape of the Box observation space
        input_dim = observation_space.shape[0]

        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)

        # Final FC layer to produce the feature dimension
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure the input is in the right shape: (batch_size, seq_len, input_dim)
        batch_size = observations.shape[0]
        observations = observations.view(
            batch_size, 1, -1
        )  # (batch_size, 1, input_dim)

        # Pass through the LSTM layer
        lstm_out, _ = self.lstm(observations)

        # Take the last LSTM output and pass through the fully connected layer
        return self.fc(lstm_out[:, -1, :])


class BoxLSTMPPOPolicy(ActorCriticPolicy):
    """
    Custom PPO policy using LSTM for Box observations.
    """

    def __init__(self, *args, **kwargs):
        features_extractor_class = BoxLSTMFeatureExtractor
        super().__init__(
            *args, **kwargs, features_extractor_class=features_extractor_class
        )


class DictLSTMPPOPolicy(ActorCriticPolicy):
    """
    Custom PPO policy using LSTM for dictionary observations.
    """

    def __init__(self, *args, **kwargs):
        features_extractor_class = DictLSTMFeatureExtractor
        super().__init__(
            *args, **kwargs, features_extractor_class=features_extractor_class
        )


class BoxLSTMMaskablePPOPolicy(MaskableActorCriticPolicy):
    """
    Custom PPO policy using LSTM for dictionary observations.
    """

    def __init__(self, *args, **kwargs):
        features_extractor_class = BoxLSTMFeatureExtractor
        super().__init__(
            *args, **kwargs, features_extractor_class=features_extractor_class
        )


class DictLSTMMaskablePPOPolicy(MaskableActorCriticPolicy):
    """
    Custom PPO policy using LSTM for dictionary observations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, features_extractor_class=DictLSTMFeatureExtractor
        )


def get_host_count_from_train_dir(train_model_dir) -> int | None:
    # Regular expression to match the number before "hosts"
    match = re.search(r"(\d+)hosts", train_model_dir)
    if not match:
        re.search(r"(\d+)nodes", train_model_dir)
    if match:
        number = match.group(1)  # Extract the matched number
        return int(number)
    return None


def datacenter_constructor(_, node) -> dict:
    return node.value


def create_logger(save_experiment, log_dir) -> sb3.common.logger.Logger:
    # log_destination = ["stdout"]
    # log_destination.extend(["csv", "tensorboard"])
    log_destination = []
    if save_experiment:
        log_destination.append("tensorboard")

    # the logger can write to stdout, csv (the file created is called progress.csv) and tensorboard
    return configure(log_dir, log_destination)


def create_callback(
    save_experiment, log_dir
) -> SaveOnBestTrainingRewardCallback | None:
    if save_experiment:
        return SaveOnBestTrainingRewardCallback(log_dir)
    # the callback writes all the .csv files and saves the model (with replay buffer) when the reward is the best
    return None


def _get_total_hosts(params):
    total_hosts = 0
    for datacenter in params["datacenters"]:
        for host_type in datacenter["hosts"]:
            total_hosts += host_type["amount"]
    return total_hosts


def compute_freeze_indices_for_multi_dc_obs(params, prev_host_count=None) -> dict:
    """
    Compute the indices and relevant parameters for freezing weights.

    Args:
        params (dict): Parameters containing host and VM configuration.
        prev_host_count (int, optional): Previous host count for transfer learning. Defaults to None.

    Returns:
        dict: A dictionary containing the computed indices and other derived values.
    """
    max_hosts = params["max_hosts"]
    cur_host_count = _get_total_hosts(params)
    max_host_pes = params["max_host_pes"]
    infr_obs_upper_bound = max_host_pes + 1
    # current infrastructure observation length
    cur_start_idx = 3 * cur_host_count * infr_obs_upper_bound
    end_idx = 3 * max_hosts * infr_obs_upper_bound

    prev_start_idx = None
    if prev_host_count is not None:
        prev_start_idx = 3 * prev_host_count * infr_obs_upper_bound

    return {
        "cur_start_idx": cur_start_idx,
        "end_idx": end_idx,
        "prev_start_idx": prev_start_idx,
    }


def compute_freeze_indices_for_tree_obs(params, prev_host_count=None) -> dict:
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
    # ------------------------
    if (
        not params["freeze_inactive_input_layer_weights"]
        or params["feature_extractor"] == "attention"
        # cannot apply index freezing with Attention extractor
    ):
        return
    if params.get("feature_extractor") == "turret":
        # In Transfer learning with GNNs, we usually freeze the GNN (representation learner)
        # and fine-tune the PPO head (decision maker).
        print("Freezing TURRET GNN Feature Extractor...")
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
        return
    if params["state_space_type"] == "tree" and params["vm_allocation_policy"] == "rl":
        indices = compute_freeze_indices_for_tree_obs(params, prev_host_count)
    elif (
        params["state_space_type"] == "dcid-dctype-freevmpes-per-host"
        and params["cloudlet_to_dc_assignment_policy"] == "rl"
    ):
        indices = compute_freeze_indices_for_multi_dc_obs(params, prev_host_count)

    if indices is None:
        return
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
    if params.get("batch_size") and params["algorithm"] != "HER":
        algorithm_kwargs["batch_size"] = params["batch_size"]
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
    if params.get("target_kl") and params["algorithm"] == "PPO":
        algorithm_kwargs["target_kl"] = params["target_kl"]
    return algorithm_kwargs


def create_correct_lstm_policy(observation_space, maskable) -> classmethod:
    if maskable:
        if isinstance(observation_space, spaces.Dict):
            return DictLSTMMaskablePPOPolicy
        return BoxLSTMMaskablePPOPolicy
    if isinstance(observation_space, spaces.Dict):
        return DictLSTMPPOPolicy
    return BoxLSTMPPOPolicy


def create_correct_policy(observation_space, params) -> str | classmethod:
    if params.get("use_lstm"):
        maskable = "Maskable" in params["algorithm"]
        return create_correct_lstm_policy(observation_space, maskable)
    if isinstance(observation_space, gym.spaces.Dict) or isinstance(
        observation_space, gym.spaces.Tuple
    ):
        return "MultiInputPolicy"  # when state is Spaces.Dict()
    return "MlpPolicy"  # when state is not Spaces.Dict()
