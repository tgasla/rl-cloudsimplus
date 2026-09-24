import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FusionMLPExtractor(BaseFeaturesExtractor):
    """
    Fusion of euromlsys's flat MLP (cross-host interaction) with type-stable
    dc_type embeddings replacing the raw scalar dc_type in each host's features.

    Per-host representation: [dc_type_emb(dc_type_emb_dim), free_pes(1)] — dc_id discarded.
    All hosts are flattened and processed jointly through a shared dense MLP, which
    preserves the cross-host interaction responsible for euromlsys's upscale transfer
    advantage, while replacing the ordinal dc_type scalar (1 < 2 < 3 is wrong) with
    independent learned vectors per infrastructure tier.

    DC type labels (cloud=1, edge=2, micro=3) are semantic infrastructure tiers
    stable across all environment variants — reviewer-safe, unlike DC ID embeddings.

    Architecture:
        infr_branch:       embed dc_type per host → flatten → Linear(17*H, hidden_dim) MLP
        job_branch:        Linear(job_obs_len, hidden_dim) MLP (same as euromlsys)
        adaptation_layer:  residual Linear(2*hidden_dim, 2*hidden_dim) for transfer
        fc:                ReLU → Linear(2*hidden_dim, features_dim)
    """

    IDX_DC_ID    = 0
    IDX_DC_TYPE  = 1
    IDX_FREE_PES = 2
    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM  = 3

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        dc_type_emb_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_dc_types: int = 3,
        use_residual: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        self.use_residual = use_residual

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))

        self.max_hosts = infr_flat // self.HOST_FEAT_DIM

        # dc_type ∈ [0, max_dc_types]: 0 = inactive/padding, 1..T = cloud/edge/micro
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_emb_dim)

        host_repr_dim = dc_type_emb_dim + 1  # type_emb + free_pes
        infr_flat_dim = host_repr_dim * self.max_hosts

        self.infr_branch = nn.Sequential(
            nn.Linear(infr_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.job_branch = nn.Sequential(
            nn.Linear(jobs_flat, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        total_dim = hidden_dim * 2
        self.adaptation_layer = nn.Linear(total_dim, total_dim)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(total_dim, features_dim),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        dc_types = host_feats[:, :, self.IDX_DC_TYPE].long().clamp(
            0, self.dc_type_embed.num_embeddings - 1
        )
        free_pes = host_feats[:, :, self.IDX_FREE_PES].unsqueeze(-1)  # [B, H, 1]

        type_embs = self.dc_type_embed(dc_types)  # [B, H, dc_type_emb_dim]
        host_repr = torch.cat([type_embs, free_pes], dim=-1)  # [B, H, emb+1]
        host_repr_flat = host_repr.view(B, -1)  # [B, H*(emb+1)]

        infr_feat = self.infr_branch(host_repr_flat)
        job_feat  = self.job_branch(jobs.view(B, -1))

        combined = torch.cat([infr_feat, job_feat], dim=-1)

        if self.use_residual:
            combined = combined + 0.1 * self.adaptation_layer(combined)

        return self.fc(combined)
