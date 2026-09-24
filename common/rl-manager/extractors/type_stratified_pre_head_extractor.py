import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TypeStratifiedPreHeadExtractor(BaseFeaturesExtractor):
    """
    Type-stratified pooling: separate (mean, max) pool per DC type.

    Root cause of transfer failure in SPANE/hybrid: pooling all DCs into one
    global vector causes distributional shift when the DC type mix changes across
    environments (e.g., cloud DC absent in Env A; 2 extra edge DCs in Env C).

    Fix: maintain one (mean, max) pair per DC type. Output is always fixed-size:
        [mean_cloud ‖ max_cloud ‖ mean_edge ‖ max_edge ‖ mean_micro ‖ max_micro]

    When a DC type is absent its bucket is a zero vector — the policy learns
    "type absent" from a stable zero, not from a shifted global mean. When extra
    DCs of an existing type appear, only that type's bucket changes, and it
    changes systematically (more capacity → higher mean, higher max).

    Pipeline:
      1. Scatter hosts → per-DC features [dc_type, sum_free_vmpes, n_active_hosts]
         (SPANE aggregation — no DC ID, no positional bias)
      2. Shared DC MLP applied independently to each DC (weight-tied)
      3. For each type t ∈ {1 … max_dc_types}:
           type_mask = dc_mask & (dc_type == t)
           mean_t  = masked mean pool(dc_embs, type_mask)    [B, dc_emb_dim]
           max_t   = masked max  pool(dc_embs, type_mask)    [B, dc_emb_dim]
           → zero vector when type is absent (stable, learnable signal)
      4. cluster_emb = cat([mean_1, max_1, …, mean_T, max_T])  [B, T*2*dc_emb_dim]
      5. Shared Job MLP + masked mean pool → mean_job_emb      [B, job_emb_dim]
      6. head(concat(cluster_emb, mean_job_emb)) → base        [B, features_dim]
      7. base + 0.1 * adaptation_layer(base)  (euromlsys residual — fast fine-tuning path)

    Config params (via features_extractor_kwargs):
      features_dim, dc_emb_dim, job_emb_dim, hidden_dim, max_dc_types, max_datacenters
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
        dc_emb_dim: int = 32,
        job_emb_dim: int = 64,
        hidden_dim: int = 128,
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

        # Shared DC MLP — weight-tied across all DCs (SPANE structural invariance)
        self.dc_mlp = nn.Sequential(
            nn.Linear(self.DC_INPUT_DIM, hidden_dim),
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
        pre_head_dim = cluster_dim + job_emb_dim
        self.head = nn.Linear(pre_head_dim, features_dim)

        # Residual adaptation layer — pre-head, full-rank linear for maximum capacity
        self.adaptation_layer = nn.Linear(pre_head_dim, pre_head_dim)

    def _aggregate_hosts_to_dc(
        self, host_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SPANE aggregation: scatter per-host observations into per-DC feature vectors.

        host_feats: [B, max_hosts, HOST_FEAT_DIM]
        Returns:
          dc_feats: [B, max_datacenters, DC_INPUT_DIM] — (mean_dc_type, sum_free_vmpes, n_hosts)
          dc_mask:  [B, max_datacenters] — True where DC has ≥1 real host
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

        # Drop slot 0 (padding accumulator)
        dc_type_acc = dc_type_acc[:, 1:]
        dc_pes_acc  = dc_pes_acc[:, 1:]
        dc_count    = dc_count[:, 1:]

        dc_mask    = dc_count > 0
        safe_count = dc_count.clamp(min=1.0)

        dc_feats = torch.stack([
            dc_type_acc / safe_count,  # mean dc_type (uniform within each DC)
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

        # dc_type is feature col 0 after aggregation (mean = constant within DC)
        dc_type_int = dc_feats[:, :, 0].round().long()                 # [B, max_dc]

        type_parts = []
        zeros = torch.zeros(B, self.dc_emb_dim, device=device)
        for t in range(1, self.max_dc_types + 1):
            type_mask   = dc_mask & (dc_type_int == t)                 # [B, max_dc]
            has_type    = type_mask.any(dim=1, keepdim=True)           # [B, 1]
            type_mask_f = type_mask.unsqueeze(-1).float()              # [B, max_dc, 1]

            # Masked mean — zero when type absent
            n_type = type_mask_f.sum(dim=1).clamp(min=1.0)            # [B, 1]
            mean_t = (dc_embs * type_mask_f).sum(dim=1) / n_type      # [B, dc_emb_dim]
            mean_t = torch.where(has_type, mean_t, zeros)

            # Masked max — fill absent slots with -inf, then restore zero when type absent
            fill = ~type_mask.unsqueeze(-1).expand_as(dc_embs)
            max_t = dc_embs.masked_fill(fill, float("-inf")).max(dim=1).values
            max_t = torch.where(has_type, max_t, zeros)

            type_parts.extend([mean_t, max_t])

        cluster_emb = torch.cat(type_parts, dim=-1)                    # [B, T*2*dc_emb_dim]

        # ── Job stream ────────────────────────────────────────────────────────
        job_mask_f   = (job_feats[:, :, 0] > 0).unsqueeze(-1).float()
        job_embs     = self.job_mlp(job_feats)
        mean_job_emb = (job_embs * job_mask_f).sum(dim=1) / job_mask_f.sum(dim=1).clamp(min=1.0)

        # ── Residual adapter (pre-head, full representation) + head ─────────────
        pre_head = torch.cat([cluster_emb, mean_job_emb], dim=-1)
        adapted = pre_head + 0.1 * self.adaptation_layer(pre_head)
        return self.head(adapted)
