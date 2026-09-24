import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RBFPoolingExtractor(BaseFeaturesExtractor):
    """
    Like HybridPoolingExtractor but replaces MultiheadAttention pooling
    with a Gaussian RBF kernel for DC token aggregation.

    Motivation: MHA on ≤8 DC tokens degenerates toward near-uniform attention weights
    (softmax over N~8 values stays near 1/N unless logits diverge strongly).
    RBF kernel avoids softmax: weight(q, k) = exp(-||q-k||² / (2σ²)).
    Weights are naturally bounded (0,1] and content-sparse — DCs whose embedding
    is far from the query in L2 space receive near-zero weight. σ is a learned
    scalar bandwidth initialised at 1.0.

    Architecture (same as hybrid except the pooling step):
      DC stream:   SPANE aggregation → shared DC MLP → RBF-weighted masked sum
      Job stream:  shared Job MLP → masked mean pool
      Output:      head(concat(cluster_emb, mean_job_emb)) → base
                   base + 0.1 * adaptation_layer(base)

    Config params (via features_extractor_kwargs):
      features_dim, dc_emb_dim, job_emb_dim, hidden_dim, max_datacenters
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
        max_datacenters: int = 8,
    ):
        super().__init__(observation_space, features_dim)

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))
        self.max_hosts       = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs        = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters

        # Shared DC MLP — weight-tied across all DCs (SPANE structural invariance)
        self.dc_mlp = nn.Sequential(
            nn.Linear(self.DC_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dc_emb_dim),
            nn.ReLU(),
        )

        # RBF pooling parameters
        self.pool_query = nn.Parameter(torch.randn(dc_emb_dim))
        # log_sigma=0 → σ=1 at init; learned during training
        self.log_sigma  = nn.Parameter(torch.zeros(1))

        # Shared Job MLP — weight-tied across all jobs
        self.job_mlp = nn.Sequential(
            nn.Linear(self.JOB_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, job_emb_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(dc_emb_dim + job_emb_dim, features_dim)

        # Residual adaptation layer — dedicated fast fine-tuning pathway
        self.adaptation_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim),
        )

    def _aggregate_hosts_to_dc(
        self, host_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B      = host_feats.shape[0]
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
        job_feats  = jobs.view(B, self.max_jobs, self.JOB_FEAT_DIM)

        # ── DC stream ─────────────────────────────────────────────────────────
        dc_feats, dc_mask = self._aggregate_hosts_to_dc(host_feats)   # [B, max_dc, 3]
        dc_embs = self.dc_mlp(dc_feats)                                # [B, max_dc, dc_emb_dim]

        # RBF pooling: weight(q, k) = exp(-||q-k||² / (2σ²))
        q        = self.pool_query.view(1, 1, -1)                      # [1, 1, dc_emb_dim]
        dist_sq  = ((q - dc_embs) ** 2).sum(-1)                       # [B, max_dc]
        sigma_sq = torch.exp(self.log_sigma * 2).clamp(min=1e-6)      # σ² = exp(2·log_σ)
        weights  = torch.exp(-dist_sq / (2.0 * sigma_sq))             # [B, max_dc] ∈ (0,1]
        weights  = weights.masked_fill(~dc_mask, 0.0)                 # suppress padding
        weights  = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)  # normalise
        cluster_emb = (weights.unsqueeze(-1) * dc_embs).sum(1)        # [B, dc_emb_dim]

        # ── Job stream ────────────────────────────────────────────────────────
        job_mask_f   = (job_feats[:, :, 0] > 0).unsqueeze(-1).float()
        job_embs     = self.job_mlp(job_feats)
        mean_job_emb = (job_embs * job_mask_f).sum(dim=1) / job_mask_f.sum(dim=1).clamp(min=1.0)

        # ── Head + residual adapter ───────────────────────────────────────────
        base = self.head(torch.cat([cluster_emb, mean_job_emb], dim=-1))
        return base + 0.1 * self.adaptation_layer(base)
