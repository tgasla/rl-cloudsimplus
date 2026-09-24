import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TypeStratifiedEmbedExtractor(BaseFeaturesExtractor):
    """
    TypeStratifiedExtractor with a learned nn.Embedding per DC type.

    The only structural change from TypeStratifiedExtractor: the raw dc_type
    scalar (a float 1.0/2.0/3.0) fed into the shared DC MLP is replaced by a
    learned nn.Embedding(max_dc_types+1, dc_type_emb_dim). Index 0 is the
    padding/inactive type; indices 1..max_dc_types correspond to cloud, edge,
    far-edge.

    Why this matters: a scalar ordinal forces the MLP to represent types on a
    line (1 < 2 < 3). A learned embedding gives each type an independent vector
    in R^dc_type_emb_dim with no geometric constraint. The embeddings are stable
    across environments (DC type IDs are the same in B, A, and C), so the type
    representation transfers without any positional DC-slot binding.

    Pipeline:
      1. Scatter hosts → per-DC: dc_type_int, sum_free_vmpes, n_active_hosts
      2. type_emb = dc_type_emb(dc_type_int)           [B, max_dc, dc_type_emb_dim]
         dc_cont  = [sum_free_vmpes, n_active_hosts]    [B, max_dc, 2]
         dc_input = cat([type_emb, dc_cont])            [B, max_dc, dc_type_emb_dim+2]
      3. Shared DC MLP applied independently to each DC (weight-tied)
      4. Per-type (mean, max) pooling — same as TypeStratifiedExtractor
      5. Shared Job MLP + masked mean pool
      6. head(concat) → base + 0.1 * adaptation_layer(base)

    Config params (via features_extractor_kwargs):
      features_dim, dc_emb_dim, job_emb_dim, hidden_dim,
      dc_type_emb_dim, max_dc_types, max_datacenters
    """

    IDX_DC_ID     = 0
    IDX_DC_TYPE   = 1
    IDX_FREE_PES  = 2
    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM  = 3

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        dc_emb_dim: int = 32,
        job_emb_dim: int = 64,
        hidden_dim: int = 128,
        dc_type_emb_dim: int = 16,
        max_dc_types: int = 3,
        max_datacenters: int = 8,
    ):
        super().__init__(observation_space, features_dim)

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))
        self.max_hosts       = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs        = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters
        self.max_dc_types    = max_dc_types
        self.dc_emb_dim      = dc_emb_dim

        # Learned DC-type embedding: index 0=padding, 1..max_dc_types=real types
        self.dc_type_emb = nn.Embedding(max_dc_types + 1, dc_type_emb_dim)

        # Shared DC MLP — input is now type_emb (dc_type_emb_dim) + 2 continuous features
        dc_mlp_input_dim = dc_type_emb_dim + 2
        self.dc_mlp = nn.Sequential(
            nn.Linear(dc_mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dc_emb_dim),
            nn.ReLU(),
        )

        # Shared Job MLP — weight-tied across all jobs
        self.job_mlp = nn.Sequential(
            nn.Linear(self.JOB_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, job_emb_dim),
            nn.ReLU(),
        )

        # Head input: (mean_t, max_t) per type → max_dc_types * 2 * dc_emb_dim
        cluster_dim = max_dc_types * 2 * dc_emb_dim
        self.head = nn.Linear(cluster_dim + job_emb_dim, features_dim)

        # Residual adaptation layer — dedicated fast fine-tuning pathway
        self.adaptation_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim),
        )

    def _aggregate_hosts_to_dc(
        self, host_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scatter per-host observations into per-DC feature vectors.

        Returns:
          dc_type_int: [B, max_datacenters] long — DC type index (0=inactive)
          dc_cont:     [B, max_datacenters, 2] float — (sum_free_vmpes, n_active_hosts)
          dc_mask:     [B, max_datacenters] bool — True where DC has ≥1 real host
        """
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

        # Integer type per DC slot; inactive slots get index 0 (padding embedding)
        dc_type_int = (dc_type_acc / safe_count).round().long().clamp(0, self.max_dc_types)
        dc_type_int = dc_type_int.masked_fill(~dc_mask, 0)

        dc_cont = torch.stack([dc_pes_acc, dc_count], dim=-1)   # [B, max_dc, 2]

        return dc_type_int, dc_cont, dc_mask

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(B, self.max_jobs, self.JOB_FEAT_DIM)

        # ── DC stream ─────────────────────────────────────────────────────────
        dc_type_int, dc_cont, dc_mask = self._aggregate_hosts_to_dc(host_feats)

        type_embs = self.dc_type_emb(dc_type_int)              # [B, max_dc, dc_type_emb_dim]
        dc_input  = torch.cat([type_embs, dc_cont], dim=-1)    # [B, max_dc, emb_dim+2]
        dc_embs   = self.dc_mlp(dc_input)                      # [B, max_dc, dc_emb_dim]

        # Per-type (mean, max) pooling — same as TypeStratifiedExtractor
        type_parts = []
        zeros = torch.zeros(B, self.dc_emb_dim, device=device)
        for t in range(1, self.max_dc_types + 1):
            type_mask   = dc_mask & (dc_type_int == t)
            has_type    = type_mask.any(dim=1, keepdim=True)
            type_mask_f = type_mask.unsqueeze(-1).float()

            n_type = type_mask_f.sum(dim=1).clamp(min=1.0)
            mean_t = (dc_embs * type_mask_f).sum(dim=1) / n_type
            mean_t = torch.where(has_type, mean_t, zeros)

            fill  = ~type_mask.unsqueeze(-1).expand_as(dc_embs)
            max_t = dc_embs.masked_fill(fill, float("-inf")).max(dim=1).values
            max_t = torch.where(has_type, max_t, zeros)

            type_parts.extend([mean_t, max_t])

        cluster_emb = torch.cat(type_parts, dim=-1)            # [B, T*2*dc_emb_dim]

        # ── Job stream ────────────────────────────────────────────────────────
        job_mask_f   = (job_feats[:, :, 0] > 0).unsqueeze(-1).float()
        job_embs     = self.job_mlp(job_feats)
        mean_job_emb = (job_embs * job_mask_f).sum(dim=1) / job_mask_f.sum(dim=1).clamp(min=1.0)

        # ── Head + residual adapter ───────────────────────────────────────────
        base = self.head(torch.cat([cluster_emb, mean_job_emb], dim=-1))
        return base + 0.1 * self.adaptation_layer(base)
