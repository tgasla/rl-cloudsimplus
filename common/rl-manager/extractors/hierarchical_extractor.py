import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HierarchicalJobDCExtractor(BaseFeaturesExtractor):
    """
    Hierarchical job-DC cross-attention with type-level DC summaries.

    Built on IDFreeAttentionExtractor but with a key change in cross-attention:
    instead of attending to per-slot DC tokens [max_dc, D], jobs attend to
    per-TYPE DC summaries [max_dc_types, D]. This gives each job semantic context
    ("what is the typical cloud DC capacity? edge DC capacity?") rather than
    slot-level positional context, making the representation more transferable.

    Architecture:
      1. Embed hosts: [dc_type_emb, free_pes] → host_proj → host_transformer
         (padding mask on inactive hosts, same as attention_idfree)
      2. Scatter hosts → DC slots by dc_id value (scatter_add, no dc_id embedding)
      3. Aggregate DC slots → TYPE summaries: mean per dc_type → [B, max_dc_types, D]
      4. Job encoding: Linear(4, hidden_dim)
      5. Cross-attention: jobs (Q) attend to TYPE summaries (K/V) — hierarchical step
      6. Global transformer over [dc_repr ‖ enriched_jobs]
      7. Learned pool query → head → features_dim
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
        self.hidden_dim      = hidden_dim

        dc_type_dim = min(8, (max_dc_types // 2) + 1)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)

        host_input_dim = dc_type_dim + 1  # type_emb + free_pes
        self.host_proj = nn.Linear(host_input_dim, hidden_dim)

        host_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.host_encoder = nn.TransformerEncoder(
            host_enc_layer, num_layers=1, enable_nested_tensor=False
        )

        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, hidden_dim)

        # Cross-attention: jobs attend to TYPE-level DC summaries
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

        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn  = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.head = nn.Linear(hidden_dim, features_dim)

    def _embed_hosts(self, host_feats: torch.Tensor) -> torch.Tensor:
        dc_types = host_feats[:, :, self.IDX_DC_TYPE].long().clamp(
            0, self.dc_type_embed.num_embeddings - 1
        )
        free_pes = host_feats[:, :, self.IDX_FREE_PES].unsqueeze(-1)
        return self.host_proj(torch.cat([self.dc_type_embed(dc_types), free_pes], dim=-1))

    def _scatter_to_dc(self, host_repr: torch.Tensor, dc_ids: torch.Tensor) -> torch.Tensor:
        B, H, D = host_repr.shape
        device = host_repr.device
        dc_repr = torch.zeros(B, self.max_datacenters + 1, D, device=device)
        counts  = torch.zeros(B, self.max_datacenters + 1, 1, device=device)
        idx = dc_ids.clamp(0, self.max_datacenters).unsqueeze(-1).expand(-1, -1, D)
        dc_repr.scatter_add_(1, idx, host_repr)
        counts.scatter_add_(
            1,
            dc_ids.clamp(0, self.max_datacenters).unsqueeze(-1),
            torch.ones(B, H, 1, device=device),
        )
        dc_repr = dc_repr / counts.clamp(min=1.0)
        return dc_repr[:, 1:], counts[:, 1:, 0]  # [B, max_dc, D], [B, max_dc]

    def _aggregate_dc_to_types(
        self,
        dc_repr: torch.Tensor,
        dc_active: torch.Tensor,
        type_per_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate per-slot DC repr into per-TYPE summaries via scatter mean.

        dc_repr:      [B, max_dc, D]
        dc_active:    [B, max_dc] bool
        type_per_slot:[B, max_dc] long — dc_type for each active DC slot

        Returns:
          type_repr: [B, max_dc_types, D]
          type_mask: [B, max_dc_types] bool
        """
        B, _, D = dc_repr.shape
        device = dc_repr.device

        type_repr = torch.zeros(B, self.max_dc_types + 1, D, device=device)
        type_cnt  = torch.zeros(B, self.max_dc_types + 1, 1, device=device)

        t_idx = type_per_slot.clamp(0, self.max_dc_types)   # [B, max_dc]
        active_f = dc_active.float().unsqueeze(-1)           # [B, max_dc, 1]

        idx_exp = t_idx.unsqueeze(-1).expand(-1, -1, D)
        type_repr.scatter_add_(1, idx_exp, dc_repr * active_f)
        type_cnt.scatter_add_(1, t_idx.unsqueeze(-1), active_f)

        type_repr = (type_repr / type_cnt.clamp(min=1.0))[:, 1:]  # [B, max_dc_types, D]
        type_mask = type_cnt[:, 1:, 0] > 0                        # [B, max_dc_types]
        return type_repr, type_mask

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(B, self.max_jobs,  self.JOB_FEAT_DIM)
        dc_ids     = host_feats[:, :, self.IDX_DC_ID].long()

        # Type per DC slot (raw dc_type from obs, not from encoder output)
        dc_ids_clamped = dc_ids.clamp(0, self.max_datacenters)
        dc_types_raw   = host_feats[:, :, self.IDX_DC_TYPE].float()
        ones_h         = torch.ones(B, self.max_hosts, device=device)

        type_acc  = torch.zeros(B, self.max_datacenters + 1, device=device)
        slot_cnt  = torch.zeros(B, self.max_datacenters + 1, device=device)
        type_acc.scatter_add_(1, dc_ids_clamped, dc_types_raw)
        slot_cnt.scatter_add_(1, dc_ids_clamped, ones_h)
        type_per_slot = (
            (type_acc / slot_cnt.clamp(min=1.0))[:, 1:].round().long().clamp(0, self.max_dc_types)
        )  # [B, max_dc]

        # Host encoding
        h = self._embed_hosts(host_feats)
        h = self.host_encoder(h, src_key_padding_mask=(dc_ids == 0))

        # Scatter to DC slots
        dc_repr, dc_slot_cnt = self._scatter_to_dc(h, dc_ids)   # [B, max_dc, D]
        dc_active = dc_slot_cnt > 0                              # [B, max_dc]

        # Type-level summaries
        type_repr, type_mask = self._aggregate_dc_to_types(dc_repr, dc_active, type_per_slot)

        # Job encoding + hierarchical cross-attention to type summaries
        j = self.job_proj(job_feats)
        j_att, _ = self.cross_attn(
            j, type_repr, type_repr,
            key_padding_mask=~type_mask,
        )
        j = self.cross_norm(j + j_att)

        # Global encoder + pool
        combined = torch.cat([dc_repr, j], dim=1)
        encoded  = self.global_encoder(combined)

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded)
        pooled = pooled.squeeze(1)

        return self.head(pooled)
