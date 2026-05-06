import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the EuroMLSys job-placement environment.

    Handles mixed observation spaces (Box, MultiDiscrete, Discrete) by:
      - embedding categorical/discrete inputs
      - projecting continuous inputs through a two-layer MLP
      - concatenating all sub-space representations
      - applying a residual adaptation layer (optionally) for transfer

    Config params (passed via features_extractor_kwargs):
      features_dim:           output dimension
      embedding_size:         max per-category embedding size
      hidden_dim:             MLP hidden dimension per sub-space
      adaptation_bottleneck:  compress adaptation layer to dim 32 before projecting back
      use_residual:           toggle residual connection (set False for ablation)
      dropout:                dropout probability in MLP layers
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
        use_residual: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        self.use_residual = use_residual
        self.extractors = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        if hidden_dims is None:
            hidden_dims = {key: hidden_dim for key in observation_space.spaces.keys()}

        for key, subspace in observation_space.spaces.items():
            current_hidden_dim = hidden_dims.get(key, hidden_dim)

            if isinstance(subspace, spaces.MultiDiscrete):
                embedding_dims = [
                    min(embedding_size, (n // 2) + 1) for n in subspace.nvec
                ]
                self.embeddings[key] = nn.ModuleList(
                    [nn.Embedding(n, dim) for n, dim in zip(subspace.nvec, embedding_dims)]
                )
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
                embedding_dim = min(embedding_size, (subspace.n // 2) + 1)
                self.extractors[key] = nn.Sequential(
                    nn.Embedding(subspace.n, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )
                total_embedding_dim += embedding_dim

            elif isinstance(subspace, spaces.Box):
                input_dim = (
                    subspace.shape[0]
                    if len(subspace.shape) == 1
                    else np.prod(subspace.shape)
                )
                self.extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_dim, current_hidden_dim),
                    activation,
                    nn.LayerNorm(current_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(current_hidden_dim, current_hidden_dim),
                    activation,
                )
                total_embedding_dim += current_hidden_dim

            elif isinstance(subspace, spaces.Tuple):
                for i, sub_subspace in enumerate(subspace.spaces):
                    self.extractors[f"{key}_{i}"] = self._create_extractor(
                        sub_subspace, current_hidden_dim
                    )
                    total_embedding_dim += current_hidden_dim

            else:
                raise ValueError(f"Unsupported observation space type: {type(subspace)}")

        if adaptation_bottleneck:
            self.adaptation_layer = nn.Sequential(
                nn.Linear(total_embedding_dim, 32),
                activation,
                nn.Linear(32, total_embedding_dim),
            )
        else:
            self.adaptation_layer = nn.Linear(total_embedding_dim, total_embedding_dim)

        self.fc = nn.Sequential(
            activation,
            nn.Linear(total_embedding_dim, features_dim),
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, observations):
        device = next(self.parameters()).device
        observations = {key: obs.to(device) for key, obs in observations.items()}

        for key, obs_val in observations.items():
            if len(obs_val.shape) == 1:
                observations[key] = obs_val.unsqueeze(0)

        embedded_features = []

        for key, extractor in self.extractors.items():
            obs_val = observations[key]

            if key in self.embeddings:
                feature_list = [
                    emb(obs_val[:, i].long())
                    for i, emb in enumerate(self.embeddings[key])
                ]
                concatenated_embeddings = torch.cat(feature_list, dim=-1)
                transformed = extractor(concatenated_embeddings)
                embedded_features.append(transformed)
            elif isinstance(extractor[0], nn.Embedding):
                embedded_features.append(extractor(obs_val.long()))
            else:
                embedded_features.append(extractor(obs_val.float()))

        final_features = torch.cat(embedded_features, dim=-1)
        if self.use_residual:
            adapted_features = final_features + 0.1 * self.adaptation_layer(final_features)
        else:
            adapted_features = final_features
        return self.fc(adapted_features)
