import gymnasium as gym
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
    Custom feature extractor that processes dictionary observations with an LSTM.
    Handles both Box (continuous) and MultiDiscrete (categorical) spaces.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # Extract spaces from the dictionary
        self.extractors = nn.ModuleDict()
        total_input_dim = 0

        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, spaces.Box):
                # Linear layer for Box space
                input_dim = subspace.shape[0]
                self.extractors[key] = nn.Linear(
                    input_dim, 32
                )  # Feature extraction for Box
                total_input_dim += 32  # Adjusted feature size

            elif isinstance(subspace, spaces.MultiDiscrete):
                # Embedding layer for MultiDiscrete space
                num_categories = subspace.nvec  # List of category sizes
                embedding_dims = [
                    min(32, (n // 2) + 1) for n in num_categories
                ]  # Dynamic embedding size

                self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(num_categories[i], embedding_dims[i])
                        for i in range(len(num_categories))
                    ]
                )

                total_input_dim += sum(embedding_dims)  # Sum of embedding sizes

        # LSTM for sequential feature processing
        self.lstm = nn.LSTM(total_input_dim, 64, batch_first=True)

        # Final fully connected layer
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        extracted_features = []

        for key, extractor in self.extractors.items():
            if key in observations:
                extracted_features.append(extractor(observations[key]))

        # Process MultiDiscrete separately using embeddings
        if hasattr(self, "embeddings"):
            categorical_features = []
            for i, embedding in enumerate(self.embeddings):
                categorical_features.append(
                    embedding(observations["jobs_waiting_state"].long()[:, i])
                )  # One-hot embedding
            extracted_features.append(torch.cat(categorical_features, dim=-1))

        # Concatenate all extracted features
        concatenated = torch.cat(extracted_features, dim=-1)

        # Make sure concatenated has the correct shape: (batch_size, seq_len, input_dim)
        # Remove unsqueeze(0) to preserve the batch size
        batch_size = observations["jobs_waiting_state"].shape[
            0
        ]  # Assuming batch size is from this observation
        concatenated = concatenated.view(
            batch_size, 1, -1
        )  # (batch_size, seq_len, input_dim)

        # LSTM expects (batch_size, seq_len, input_dim), so we pass it directly
        lstm_out, _ = self.lstm(concatenated)

        # Take the last output of LSTM and pass it through FC layer
        return self.fc(lstm_out[:, -1, :])


class CustomLSTMPPOPolicy(MaskableActorCriticPolicy):
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
        policy = CustomLSTMPPOPolicy

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
