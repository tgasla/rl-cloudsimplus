import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PMAPoolingExtractor(BaseFeaturesExtractor):
    """
    Pooling by Multihead Attention (PMA) — Set Transformer, Lee et al. 2019.

    Replaces hybrid's single learned pool query with max_dc_types independent query
    vectors. Each query attends over all DC tokens and self-organises into a cluster
    (e.g. one discovers cloud-type load patterns, another edge-type patterns) without
    requiring dc_type labels. The max_dc_types query outputs are concatenated before
    the head, giving a structured multi-slot infrastructure summary.

    DC stream (same scatter aggregation as hybrid — no DC ID):
      1. Scatter hosts → per-DC features [dc_type, sum_free_vmpes, n_active_hosts]
      2. Shared DC MLP (weight-tied)
      3. PMA: max_dc_types queries × MHA over DC tokens with padding mask
      4. Flatten query outputs → [B, max_dc_types * dc_emb_dim]

    Job stream:
      5. Shared Job MLP → masked mean pool → [B, job_emb_dim]

    Output:
      6. head(concat) → base + 0.1 * adaptation_layer(base)
    """

    IDX_DC_ID     = 0
    IDX_DC_TYPE   = 1
    IDX_FREE_PES  = 2
    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM  = 3
    DC_INPUT_DIM  = 3  # [dc_type, sum_free_vmpes, n_active_hosts]

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        dc_emb_dim: int = 64,
        job_emb_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_datacenters: int = 8,
        max_dc_types: int = 3,
    ):
        super().__init__(observation_space, features_dim)

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))
        self.max_hosts = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs  = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters
        self.max_dc_types    = max_dc_types

        self.dc_mlp = nn.Sequential(
            nn.Linear(self.DC_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dc_emb_dim),
            nn.ReLU(),
        )

        # PMA: max_dc_types independent query vectors
        self.pool_queries = nn.Parameter(torch.randn(1, max_dc_types, dc_emb_dim))
        self.pool_attn = nn.MultiheadAttention(
            dc_emb_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.job_mlp = nn.Sequential(
            nn.Linear(self.JOB_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, job_emb_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(max_dc_types * dc_emb_dim + job_emb_dim, features_dim)

        self.adaptation_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim),
        )

    def _aggregate_hosts_to_dc(
        self, host_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = host_feats.shape[0]
        device = host_feats.device

        dc_ids   = host_feats[:, :, self.IDX_DC_ID].long().clamp(0, self.max_datacenters)
        dc_types = host_feats[:, :, self.IDX_DC_TYPE]
        free_pes = host_feats[:, :, self.IDX_FREE_PES]
        ones     = torch.ones(B, self.max_hosts, device=device)

        dc_type_acc = torch.zeros(B, self.max_datacenters + 1, device=device)
        dc_pes_acc  = torch.zeros(B, self.max_datacenters + 1, device=device)
        dc_count    = torch.zeros(B, self.max_datacenters + 1, device=device)

        dc_type_acc.scatter_add_(1, dc_ids, dc_types)
        dc_pes_acc.scatter_add_(1, dc_ids, free_pes)
        dc_count.scatter_add_(1, dc_ids, ones)

        dc_type_acc = dc_type_acc[:, 1:]
        dc_pes_acc  = dc_pes_acc[:, 1:]
        dc_count    = dc_count[:, 1:]

        dc_mask    = dc_count > 0
        safe_count = dc_count.clamp(min=1.0)

        dc_feats = torch.stack([
            dc_type_acc / safe_count,
            dc_pes_acc,
            dc_count,
        ], dim=-1)

        return dc_feats, dc_mask

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(B, self.max_jobs,  self.JOB_FEAT_DIM)

        # DC stream
        dc_feats, dc_mask = self._aggregate_hosts_to_dc(host_feats)
        dc_embs = self.dc_mlp(dc_feats)                                # [B, max_dc, dc_emb_dim]

        queries = self.pool_queries.expand(B, -1, -1)                  # [B, max_dc_types, dc_emb_dim]
        cluster, _ = self.pool_attn(
            queries, dc_embs, dc_embs, key_padding_mask=~dc_mask
        )                                                               # [B, max_dc_types, dc_emb_dim]
        cluster_emb = cluster.reshape(B, -1)                           # [B, max_dc_types * dc_emb_dim]

        # Job stream
        job_mask_f = (job_feats[:, :, 0] > 0).unsqueeze(-1).float()
        job_embs   = self.job_mlp(job_feats)
        mean_job   = (job_embs * job_mask_f).sum(1) / job_mask_f.sum(1).clamp(min=1.0)

        base = self.head(torch.cat([cluster_emb, mean_job], dim=-1))
        return base + 0.1 * self.adaptation_layer(base)
