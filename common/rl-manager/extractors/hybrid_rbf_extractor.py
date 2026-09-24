import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HybridRBFPoolExtractor(BaseFeaturesExtractor):
    """
    HybridPoolingExtractor with RBF kernel replacing MHA for DC pooling.

    On small token sets (≤8 DC slots), standard softmax attention produces near-uniform
    weights (~1/n each), providing little selectivity. The RBF kernel replaces the
    dot-product attention with a Gaussian similarity:

        w_i = exp(-||q - k_i||^2 / (2 * sigma^2))

    where q is the learned pool query vector and k_i is DC token i. This gives genuine
    locality: DC tokens similar to the query attract; dissimilar tokens are suppressed.
    The bandwidth sigma is a learned scalar (exp(log_sigma)) shared across all DCs.

    Architecture identical to HybridPoolingExtractor except:
    - pool_attn (nn.MultiheadAttention) replaced by RBF weighted sum
    - pool_query: [dc_emb_dim] (1D, no head dimension)
    - log_sigma: scalar learnable parameter (initialized to 0 → sigma=1)
    - No n_heads parameter needed
    """

    IDX_DC_ID     = 0
    IDX_DC_TYPE   = 1
    IDX_FREE_PES  = 2
    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM  = 3
    DC_INPUT_DIM  = 3

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        dc_emb_dim: int = 64,
        job_emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_datacenters: int = 8,
    ):
        super().__init__(observation_space, features_dim)

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))
        self.max_hosts       = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs        = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters

        self.dc_mlp = nn.Sequential(
            nn.Linear(self.DC_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dc_emb_dim),
            nn.ReLU(),
        )

        # RBF pooling: query vector + learnable bandwidth
        self.pool_query = nn.Parameter(torch.randn(1, dc_emb_dim))
        self.log_sigma  = nn.Parameter(torch.zeros(1))

        self.job_mlp = nn.Sequential(
            nn.Linear(self.JOB_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, job_emb_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(dc_emb_dim + job_emb_dim, features_dim)

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

    def _rbf_pool(
        self, dc_embs: torch.Tensor, dc_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        RBF-weighted mean pool over DC tokens.

        dc_embs: [B, max_dc, dc_emb_dim]
        dc_mask: [B, max_dc] — True for active DCs
        Returns: [B, dc_emb_dim]
        """
        q     = self.pool_query.expand(dc_embs.shape[0], -1)       # [B, dc_emb_dim]
        sigma = self.log_sigma.exp().clamp(min=1e-3)

        diffs       = dc_embs - q.unsqueeze(1)                     # [B, max_dc, dc_emb_dim]
        rbf_weights = (-0.5 * (diffs ** 2).sum(-1) / sigma ** 2).exp()  # [B, max_dc]
        rbf_weights = rbf_weights.masked_fill(~dc_mask, 0.0)
        rbf_weights = rbf_weights / rbf_weights.sum(-1, keepdim=True).clamp(min=1e-6)

        return (rbf_weights.unsqueeze(-1) * dc_embs).sum(1)        # [B, dc_emb_dim]

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(B, self.max_jobs,  self.JOB_FEAT_DIM)

        # DC stream
        dc_feats, dc_mask = self._aggregate_hosts_to_dc(host_feats)
        dc_embs    = self.dc_mlp(dc_feats)
        cluster_emb = self._rbf_pool(dc_embs, dc_mask)             # [B, dc_emb_dim]

        # Job stream
        job_mask_f = (job_feats[:, :, 0] > 0).unsqueeze(-1).float()
        job_embs   = self.job_mlp(job_feats)
        mean_job   = (job_embs * job_mask_f).sum(1) / job_mask_f.sum(1).clamp(min=1.0)

        base = self.head(torch.cat([cluster_emb, mean_job], dim=-1))
        return base + 0.1 * self.adaptation_layer(base)
