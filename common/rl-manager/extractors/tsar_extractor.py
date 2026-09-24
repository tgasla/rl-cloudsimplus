import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TSARExtractor(BaseFeaturesExtractor):
    """
    TSAR: Type-Stratified Adaptive Representation.

    Designed from the empirical benchmark to dominate both C→A (DC type disappears)
    and C→B (fewer DCs, all types present) transfer scenarios. It closes three
    specific gaps that prevent HierarchicalJobDCExtractor from winning C→B while
    keeping its C→A-winning TYPE-level cross-attention intact.

    Benchmark motivation:
      - HierarchicalJobDCExtractor wins C→A (absolute AUC #1: 3.306) because
        TYPE-level cross-attention cleanly handles absent DC types: the missing
        type's token is masked → zero attention weight → no signal contamination.
      - Yet hierarchical is only #4 on C→B (AUC 10.492) due to three fixable gaps:
        (a) no src_key_padding_mask on global_encoder — inactive DC slot tokens
            (zero vectors) freely attend to real tokens, introducing noise;
        (b) standard attention pool — no locality bias; absent types' near-zero
            encoder outputs still receive 1/N softmax weight;
        (c) no adaptation layer — euromlsys and ARIA outperform on C→B partly
            because ~6k adaptation params provide a dedicated fine-tuning pathway.
      - TSAR closes all three gaps while keeping the TYPE-level cross-attention.

    Components:

      1. TYPE-level cross-attention (from HierarchicalJobDCExtractor — C→A winner)
         Jobs attend to per-TYPE DC summaries [max_dc_types, D] rather than per-DC
         tokens [max_dc, D]. When a DC type is absent, its type token is zero AND
         masked (key_padding_mask=True) → strict zero contribution to job embeddings.
         This is more transfer-stable than DC-level cross-attention (ARIA) where
         individual DC tokens vanish but the attention over remaining tokens is
         unnormalised relative to the training distribution.

      2. Masked global encoder (correction over HierarchicalJobDCExtractor)
         src_key_padding_mask covers inactive DC slots AND inactive job slots.
         Without this mask, ~5 zero-valued padding-slot tokens in [dc_repr ‖ j]
         attend to real tokens in every training step, diluting representations.
         The mask also ensures inactive DC slots do not receive gradient updates
         from the global encoder, preventing spurious parameter specialisation.

      3. RBF pool at final step (from ARIAExtractor)
         w_i = exp(-||q - encoded_i||² / 2σ²), masked to valid tokens only, then
         renormalised. Inactive DC slots (near-zero after masked global encoder) get
         near-zero RBF weight — a second filter after the padding mask. Works
         synergistically with the TYPE-level cross-attention: types that contributed
         nothing to job embeddings are also suppressed in the final pool.
         σ is a learned scalar (init log_sigma=0 → σ=1, equal initial weights).

      4. Residual adaptation layer (from euromlsys / ARIAExtractor)
         base + 0.1 × adapter(base), bottleneck features_dim → features_dim//2 →
         features_dim. Provides a dedicated low-rank fine-tuning pathway during
         transfer without overriding the pre-trained representation. The 0.1 scale
         preserves zero-shot performance while giving gradients a clear route to
         specialise. Closes the C→B capacity gap empirically attributed to this
         component across the benchmark.

    Architecture pipeline:
      host_encoder → scatter_to_dc → aggregate_dc_to_types
      → cross_attn(Q=jobs, K/V=type_repr, key_padding_mask=~type_mask)
      → global_encoder([dc_repr ‖ enriched_jobs], src_key_padding_mask=~active)
      → RBF_pool(valid_mask = dc_active ‖ job_active)
      → head → base + 0.1 × adaptation_layer(base)

    Transfer invariances:
      Permutation : scatter_add keyed on dc_id VALUE; cross_attn/global_encoder
                   are set-equivariant.
      Count       : RBF valid_mask zeros inactive DC/job slots; padding mask
                   prevents inactive tokens from polluting active representations.
      Type        : dc_type embedded via nn.Embedding (no ordinal assumption);
                   type-level aggregation means absent types → zero+masked token.
      Workload    : TYPE-level cross-attention lets each job query infrastructure
                   at type granularity → stable when DC counts within a type change.
      Scale       : valid_mask covers active DCs + active jobs; no operation
                   depends on total token count or sequence length.

    Pros:
      - TYPE-level cross-attention is strictly more transfer-stable than DC-level
        when DC types disappear (supported empirically by hierarchical's C→A win).
      - Masked global encoder prevents noise injection from zero-padding slots.
      - RBF pool + adaptation layer close the C→B gap vs. hierarchical.
      - All four components have independent empirical validation in the benchmark.

    Cons:
      - Slightly more complex than hierarchical (~6k extra params for adaptation).
      - TYPE-level aggregation loses within-type DC diversity before cross-attn
        (e.g., two cloud DCs with very different loads are averaged before jobs
        query them). Per-type attention pooling (PMA-style) would fix this but
        adds parameters.
      - Empirical validation required; the hypothesis is benchmark-motivated but
        not yet confirmed by a completed training run.
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
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_datacenters: int = 8,
        max_dc_types: int = 3,
    ):
        super().__init__(observation_space, features_dim)

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))
        self.max_hosts       = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs        = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters
        self.max_dc_types    = max_dc_types

        dc_type_dim = min(8, (max_dc_types // 2) + 1)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)
        self.host_proj = nn.Linear(dc_type_dim + 1, hidden_dim)

        host_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.host_encoder = nn.TransformerEncoder(
            host_enc_layer, num_layers=1, enable_nested_tensor=False
        )

        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, hidden_dim)

        # Cross-attention: jobs query TYPE-level DC summaries (not per-slot DC tokens).
        # Absent types are masked → zero contribution; stable when a type disappears.
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # RBF pool: learned query + bandwidth. log_sigma=0 → σ=1 (equal initial weights).
        self.pool_query = nn.Parameter(torch.randn(hidden_dim))
        self.log_sigma  = nn.Parameter(torch.zeros(1))

        self.head = nn.Linear(hidden_dim, features_dim)

        # Residual adaptation: bottleneck prevents adapter from overriding base repr.
        self.adaptation_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim),
        )

    def _embed_hosts(self, host_feats: torch.Tensor) -> torch.Tensor:
        dc_types = host_feats[:, :, self.IDX_DC_TYPE].long().clamp(
            0, self.dc_type_embed.num_embeddings - 1
        )
        free_pes = host_feats[:, :, self.IDX_FREE_PES].unsqueeze(-1)
        return self.host_proj(torch.cat([self.dc_type_embed(dc_types), free_pes], dim=-1))

    def _scatter_to_dc(
        self, host_repr: torch.Tensor, dc_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter-mean host representations to DC slots; returns (dc_repr, dc_active)."""
        B, H, D = host_repr.shape
        device = host_repr.device
        clamped = dc_ids.clamp(0, self.max_datacenters)
        dc_repr = torch.zeros(B, self.max_datacenters + 1, D, device=device)
        counts  = torch.zeros(B, self.max_datacenters + 1, 1, device=device)
        dc_repr.scatter_add_(1, clamped.unsqueeze(-1).expand(-1, -1, D), host_repr)
        counts.scatter_add_(1, clamped.unsqueeze(-1), torch.ones(B, H, 1, device=device))
        dc_repr = dc_repr / counts.clamp(min=1.0)
        dc_active = counts[:, 1:, 0] > 0   # [B, max_dc]
        return dc_repr[:, 1:], dc_active

    def _get_type_per_slot(
        self, host_feats: torch.Tensor, dc_ids: torch.Tensor
    ) -> torch.Tensor:
        """Derive dc_type for each DC slot from host features (all hosts in a DC share one type)."""
        B = host_feats.shape[0]
        device = host_feats.device
        clamped = dc_ids.clamp(0, self.max_datacenters)
        dc_types_raw = host_feats[:, :, self.IDX_DC_TYPE].float()
        type_acc = torch.zeros(B, self.max_datacenters + 1, device=device)
        slot_cnt = torch.zeros(B, self.max_datacenters + 1, device=device)
        type_acc.scatter_add_(1, clamped, dc_types_raw)
        slot_cnt.scatter_add_(1, clamped, torch.ones(B, self.max_hosts, device=device))
        return (
            (type_acc / slot_cnt.clamp(min=1.0))[:, 1:].round().long().clamp(0, self.max_dc_types)
        )  # [B, max_dc]

    def _aggregate_dc_to_types(
        self,
        dc_repr: torch.Tensor,
        dc_active: torch.Tensor,
        type_per_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter-mean DC slot representations into per-TYPE summaries."""
        B, _, D = dc_repr.shape
        device = dc_repr.device
        type_repr = torch.zeros(B, self.max_dc_types + 1, D, device=device)
        type_cnt  = torch.zeros(B, self.max_dc_types + 1, 1, device=device)
        t_idx    = type_per_slot.clamp(0, self.max_dc_types)
        active_f = dc_active.float().unsqueeze(-1)
        type_repr.scatter_add_(1, t_idx.unsqueeze(-1).expand(-1, -1, D), dc_repr * active_f)
        type_cnt.scatter_add_(1, t_idx.unsqueeze(-1), active_f)
        type_repr = (type_repr / type_cnt.clamp(min=1.0))[:, 1:]  # [B, max_dc_types, D]
        type_mask = type_cnt[:, 1:, 0] > 0                        # [B, max_dc_types]
        return type_repr, type_mask

    def _rbf_pool(
        self, encoded: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """RBF-weighted pool over valid tokens; zero weight for masked-out tokens."""
        q = self.pool_query.view(1, 1, -1)
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        rbf_weights = (-0.5 * ((encoded - q) ** 2).sum(-1) / sigma ** 2).exp()  # [B, T]
        rbf_weights = rbf_weights.masked_fill(~valid_mask, 0.0)
        rbf_weights = rbf_weights / rbf_weights.sum(-1, keepdim=True).clamp(min=1e-6)
        return (rbf_weights.unsqueeze(-1) * encoded).sum(1)  # [B, D]

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(B, self.max_jobs,  self.JOB_FEAT_DIM)
        dc_ids = host_feats[:, :, self.IDX_DC_ID].long()

        h = self._embed_hosts(host_feats)
        h = self.host_encoder(h, src_key_padding_mask=(dc_ids == 0))

        dc_repr, dc_active = self._scatter_to_dc(h, dc_ids)
        type_per_slot = self._get_type_per_slot(host_feats, dc_ids)
        type_repr, type_mask = self._aggregate_dc_to_types(dc_repr, dc_active, type_per_slot)

        j = self.job_proj(job_feats)
        # TYPE-level cross-attention: absent DC types are masked → zero job-context contribution
        j_att, _ = self.cross_attn(j, type_repr, type_repr, key_padding_mask=~type_mask)
        j = self.cross_norm(j + j_att)

        job_active = job_feats[:, :, 0] > 0
        padding_mask = torch.cat([~dc_active, ~job_active], dim=1)  # True = ignore (PyTorch)
        combined = torch.cat([dc_repr, j], dim=1)
        encoded  = self.global_encoder(combined, src_key_padding_mask=padding_mask)

        valid_mask = torch.cat([dc_active, job_active], dim=1)
        pooled = self._rbf_pool(encoded, valid_mask)

        base = self.head(pooled)
        return base + 0.1 * self.adaptation_layer(base)
