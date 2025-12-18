import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for handling mixed observation spaces (Discrete, MultiDiscrete, Box, Tuple).
    Supports an optional adaptation bottleneck for transfer learning.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        embedding_size: int = 32,
        hidden_dim: int = 128,
        hidden_dims: dict = None,
        activation: nn.Module = nn.ReLU(),
        adaptation_bottleneck: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize the feature extractor.

        Args:
            observation_space (spaces.Dict): The observation space (a dictionary of subspaces).
            features_dim (int): The dimension of the output feature vector.
            embedding_size (int): The size of embeddings for discrete and multi-discrete spaces.
            hidden_dim (int): The default hidden dimension for MLPs.
            hidden_dims (dict): Custom hidden dimensions per key in the observation space.
            activation (nn.Module): The activation function to use.
            adaptation_bottleneck (bool): Whether to use a bottleneck layer for adaptation.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()  # Separate embedding layers for MultiDiscrete
        total_embedding_dim = 0

        if hidden_dims is None:
            hidden_dims = {key: hidden_dim for key in observation_space.spaces.keys()}

        for key, subspace in observation_space.spaces.items():
            current_hidden_dim = hidden_dims.get(key, hidden_dim)

            if isinstance(subspace, spaces.MultiDiscrete):
                # Create embeddings separately for each dimension in MultiDiscrete
                embedding_dims = [
                    min(embedding_size, (n // 2) + 1) for n in subspace.nvec
                ]
                self.embeddings[key] = nn.ModuleList(
                    [
                        nn.Embedding(n, dim)
                        for n, dim in zip(subspace.nvec, embedding_dims)
                    ]
                )

                # MLP for processing concatenated embeddings
                input_dim = sum(embedding_dims)
                self.extractors[key] = nn.Sequential(
                    nn.Linear(input_dim, current_hidden_dim),
                    activation,
                    nn.LayerNorm(current_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(current_hidden_dim, current_hidden_dim),
                    activation,
                )
                total_embedding_dim += current_hidden_dim

            elif isinstance(subspace, spaces.Discrete):
                # Embedding for Discrete spaces
                embedding_dim = min(embedding_size, (subspace.n // 2) + 1)
                self.extractors[key] = nn.Sequential(
                    nn.Embedding(subspace.n, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )
                total_embedding_dim += embedding_dim

            elif isinstance(subspace, spaces.Box):
                # MLP for Box (continuous) spaces
                input_dim = (
                    subspace.shape[0]
                    if len(subspace.shape) == 1
                    else np.prod(subspace.shape)
                )
                self.extractors[key] = nn.Sequential(
                    nn.Flatten(),  # Flatten if the input is multi-dimensional
                    nn.Linear(input_dim, current_hidden_dim),
                    activation,
                    nn.LayerNorm(current_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(current_hidden_dim, current_hidden_dim),
                    activation,
                )
                total_embedding_dim += current_hidden_dim

            elif isinstance(subspace, spaces.Tuple):
                # Recursively process each element in the tuple
                for i, sub_subspace in enumerate(subspace.spaces):
                    self.extractors[f"{key}_{i}"] = self._create_extractor(
                        sub_subspace, current_hidden_dim
                    )
                    total_embedding_dim += current_hidden_dim

            else:
                raise ValueError(
                    f"Unsupported observation space type: {type(subspace)}"
                )

        # Adaptation layer for transfer learning
        if adaptation_bottleneck:
            self.adaptation_layer = nn.Sequential(
                nn.Linear(total_embedding_dim, 32),
                activation,
                nn.Linear(32, total_embedding_dim),
            )
        else:
            self.adaptation_layer = nn.Linear(total_embedding_dim, total_embedding_dim)

        # Final projection layer
        self.fc = nn.Sequential(
            activation,
            nn.Linear(total_embedding_dim, features_dim),
        )

        # Initialize weights
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """Initialize weights using Xavier initialization and zero biases."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, observations):
        """
        Forward pass through the feature extractor.

        Args:
            observations (Dict[str, torch.Tensor]): A dictionary of observations.

        Returns:
            torch.Tensor: The extracted features.
        """
        device = next(self.parameters()).device
        observations = {
            key: obs.to(device) for key, obs in observations.items()
        }  # Move all observations to the device

        # Ensure batch dimension exists
        for key, obs_val in observations.items():
            if len(obs_val.shape) == 1:  # No batch dimension
                observations[key] = obs_val.unsqueeze(0)  # Add batch dimension

        embedded_features = []

        for key, extractor in self.extractors.items():
            obs_val = observations[key]

            if key in self.embeddings:  # MultiDiscrete case
                # Apply embeddings to each dimension and concatenate
                feature_list = [
                    emb(obs_val[:, i].long())
                    for i, emb in enumerate(self.embeddings[key])
                ]
                concatenated_embeddings = torch.cat(feature_list, dim=-1)
                transformed = extractor(concatenated_embeddings)  # Apply MLP
                embedded_features.append(transformed)

            elif isinstance(extractor[0], nn.Embedding):  # Discrete case
                embedded_features.append(extractor(obs_val.long()))

            else:  # Box or Tuple case
                embedded_features.append(extractor(obs_val.float()))

        # Concatenate all features
        final_features = torch.cat(embedded_features, dim=-1)

        # Apply residual adaptation
        adapted_features = final_features + 0.1 * self.adaptation_layer(final_features)

        # Final projection
        return self.fc(adapted_features)
