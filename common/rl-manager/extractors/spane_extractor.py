import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SPANEFeatureExtractor(BaseFeaturesExtractor):
    """
    SPANE-inspired symmetry-preserving feature extractor for DC-level job placement.

    Adapts the embedding-generation module from:
      "Symmetry-Preserving Architecture for Multi-NUMA Environments (SPANE)",
      Chan Tin Ping et al., IEEE INFOCOM 2025.

    Core property: a shared MLP (weight-tied across all DCs) + mean pooling produces
    a cluster embedding that is (a) permutation-invariant to DC ordering and
    (b) count-invariant across environments with different numbers of DCs.
    This structural invariance is what allows SPANE to transfer across cluster sizes
    without retraining — the mean of N embeddings and the mean of M embeddings live
    in the same space regardless of N vs M.

    Pipeline:
      1. Scatter host observations into per-DC feature vectors:
           [dc_type (float), sum_free_vmpes, n_active_hosts]
      2. Shared DC MLP — same weights applied to every DC independently.
      3. Mean pool over real (non-padding) DCs → cluster_emb.
      4. Shared Job MLP — same weights applied to every waiting job independently.
      5. Mean pool over real (non-padding) jobs → mean_job_emb.
      6. head(concat(cluster_emb, mean_job_emb)) → features_dim output.

    Config params (via features_extractor_kwargs):
      features_dim:    output embedding dimension
      dc_emb_dim:      DC embedding dimension (output of DC MLP)
      job_emb_dim:     job embedding dimension (output of Job MLP)
      hidden_dim:      hidden width of both MLPs
      max_datacenters: max DCs in any environment (sets scatter buffer size)
    """

    IDX_DC_ID    = 0
    IDX_DC_TYPE  = 1
    IDX_FREE_PES = 2
    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM  = 4
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

        self.max_hosts = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs  = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters

        # Shared DC MLP — weight-tied across all DCs (the symmetry-preserving kernel)
        self.dc_mlp = nn.Sequential(
            nn.Linear(self.DC_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dc_emb_dim),
            nn.ReLU(),
        )

        # Shared Job MLP — weight-tied across all waiting jobs
        self.job_mlp = nn.Sequential(
            nn.Linear(self.JOB_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, job_emb_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(dc_emb_dim + job_emb_dim, features_dim),
            nn.ReLU(),
        )

    def _aggregate_hosts_to_dc(
        self, host_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Scatter per-host observations into per-DC feature vectors.

        host_feats: [B, max_hosts, HOST_FEAT_DIM]  — (dc_id, dc_type, free_vmpes)

        Returns:
          dc_feats: [B, max_datacenters, DC_INPUT_DIM]
                      columns: (mean_dc_type, sum_free_vmpes, n_active_hosts)
          dc_mask:  [B, max_datacenters]  — True where DC has ≥1 real host
        """
        B = host_feats.shape[0]
        device = host_feats.device

        dc_ids   = host_feats[:, :, self.IDX_DC_ID].long().clamp(0, self.max_datacenters)
        dc_types = host_feats[:, :, self.IDX_DC_TYPE]
        free_pes = host_feats[:, :, self.IDX_FREE_PES]
        ones     = torch.ones(B, self.max_hosts, device=device)

        # Accumulate into slots 0..max_datacenters (slot 0 = padding accumulator)
        dc_type_acc = torch.zeros(B, self.max_datacenters + 1, device=device)
        dc_pes_acc  = torch.zeros(B, self.max_datacenters + 1, device=device)
        dc_count    = torch.zeros(B, self.max_datacenters + 1, device=device)

        dc_type_acc.scatter_add_(1, dc_ids, dc_types)
        dc_pes_acc.scatter_add_(1, dc_ids, free_pes)
        dc_count.scatter_add_(1, dc_ids, ones)

        # Drop slot 0 (padding accumulator) → [B, max_datacenters]
        dc_type_acc = dc_type_acc[:, 1:]
        dc_pes_acc  = dc_pes_acc[:, 1:]
        dc_count    = dc_count[:, 1:]

        dc_mask    = dc_count > 0
        safe_count = dc_count.clamp(min=1.0)

        # [B, max_datacenters, DC_INPUT_DIM]
        dc_feats = torch.stack([
            dc_type_acc / safe_count,  # mean dc_type (same for all hosts in DC)
            dc_pes_acc,                # total free vmpes in DC
            dc_count,                  # number of active hosts (DC scale indicator)
        ], dim=-1)

        return dc_feats, dc_mask

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        batch_size = infr.shape[0]

        host_feats = infr.view(batch_size, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(batch_size, self.max_jobs,  self.JOB_FEAT_DIM)

        # ── DC stream: shared MLP → mean pool ────────────────────────────────
        dc_feats, dc_mask = self._aggregate_hosts_to_dc(host_feats)
        dc_embs  = self.dc_mlp(dc_feats)                            # [B, max_dc, dc_emb_dim]
        dc_mask_f = dc_mask.unsqueeze(-1).float()                   # [B, max_dc, 1]
        n_real_dcs = dc_mask_f.sum(dim=1).clamp(min=1.0)           # [B, 1]
        cluster_emb = (dc_embs * dc_mask_f).sum(dim=1) / n_real_dcs  # [B, dc_emb_dim]

        # ── Job stream: shared MLP → mean pool ───────────────────────────────
        # Padding jobs have cores == 0
        job_mask_f = (job_feats[:, :, 0] > 0).unsqueeze(-1).float()  # [B, max_jobs, 1]
        job_embs   = self.job_mlp(job_feats)                          # [B, max_jobs, job_emb_dim]
        n_real_jobs = job_mask_f.sum(dim=1).clamp(min=1.0)           # [B, 1]
        mean_job_emb = (job_embs * job_mask_f).sum(dim=1) / n_real_jobs  # [B, job_emb_dim]

        # ── Output ────────────────────────────────────────────────────────────
        return self.head(torch.cat([cluster_emb, mean_job_emb], dim=-1))
