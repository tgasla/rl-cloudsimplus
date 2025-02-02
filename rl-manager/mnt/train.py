from pyexpat import features
import gymnasium as gym
from numpy import isin
import gym_cloudsimplus  # noqa: F401
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from gymnasium import spaces
import torch
from torch import nn

from utils.misc import (
    create_logger,
    create_callback,
    maybe_freeze_weights,
    vectorize_env,
    get_suitable_device,
    get_algorithm,
    create_kwargs_with_algorithm_params,
    create_policy_from_obs_space_type,
)


class DictLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Generalized feature extractor for dictionary observations with Box and MultiDiscrete spaces.
    Uses LSTM for sequential processing.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()  # Store embeddings for MultiDiscrete keys
        total_input_dim = 0

        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, spaces.Box):
                # Linear layer for Box space
                input_dim = subspace.shape[0]
                self.extractors[key] = nn.Linear(input_dim, 32)
                total_input_dim += 32

            elif isinstance(subspace, spaces.MultiDiscrete):
                # Create embeddings for each MultiDiscrete key separately
                num_categories = subspace.nvec
                embedding_dims = [min(32, (n // 2) + 1) for n in num_categories]

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


class BoxLSTMPPOPolicy(MaskableActorCriticPolicy):
    """
    Custom PPO policy using LSTM for dictionary observations.
    """

    def __init__(self, *args, **kwargs):
        features_extractor_class = BoxLSTMFeatureExtractor
        super().__init__(
            *args, **kwargs, features_extractor_class=features_extractor_class
        )


class DictLSTMPPOPolicy(MaskableActorCriticPolicy):
    """
    Custom PPO policy using LSTM for dictionary observations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, features_extractor_class=DictLSTMFeatureExtractor
        )


def train(params, jobs):
    # Select the appropriate algorithm
    algorithm = get_algorithm(params["algorithm"], params["vm_allocation_policy"])

    # Create and wrap the environment
    env = gym.make("SingleDC-v0", params=params, jobs=jobs)
    # Monitor needs the environment to have a render_mode set
    # If render_mode is None, it will give a warning.
    #   add info_keywords if needed
    # If log_dir is None, it will not log anything
    env = Monitor(env, params.get("log_dir"))
    env = vectorize_env(env, params["algorithm"])

    device = get_suitable_device(params["algorithm"])

    algorithm_kwargs = create_kwargs_with_algorithm_params(env, params)

    policy = create_policy_from_obs_space_type(env.observation_space)
    if params["use_lstm"]:
        if isinstance(env.observation_space, spaces.Dict):
            policy = DictLSTMPPOPolicy
        elif isinstance(env.observation_space, spaces.Box):
            policy = BoxLSTMPPOPolicy

    # Instantiate the agent
    model = algorithm(policy=policy, env=env, device=device, **algorithm_kwargs)
    maybe_freeze_weights(model, params)

    callback = create_callback(params["save_experiment"], params["log_dir"])
    logger = create_logger(params["save_experiment"], params["log_dir"])
    model.set_logger(logger)

    # Train the agent
    model.learn(total_timesteps=params["timesteps"], log_interval=1, callback=callback)

    # Close the environment and free the resources
    env.close()

    # Delete the model from memory
    del model
