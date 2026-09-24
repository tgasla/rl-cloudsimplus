import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


_STRUCT_DIM = 5  # n_dcs, n_cloud, n_edge, n_micro, total_free_pes (all normalized to [0,1])


class SWATExtractor(BaseFeaturesExtractor):
    """
    Structure-Aware Transformer for Inhomogeneous Multi-Task RL.

    Extends IDFreeAttentionExtractor with an explicit topology descriptor token
    prepended to the global encoder's input sequence. The token encodes the
    current environment's topology (how many DCs, type composition, load density)
    computed fresh from each observation — no lookup, no DC ID dependency.

    The agent can distinguish "Env A: 2 DCs, edge-only" from "Env C: 5 DCs,
    mixed cloud/edge/micro" and tune its attention accordingly, without
    sacrificing permutation invariance.

    Architecture:
      1. Embed hosts: [dc_type_emb, free_pes] → host_proj → host_transformer
         (same as IDFreeAttentionExtractor; padding mask on inactive hosts)
      2. Scatter hosts → DC slots [B, max_dc, D]; compute dc_mask [B, max_dc]
      3. Topology descriptor [B, 5]:
         [n_active_dcs/max_dc, n_cloud/max_dc, n_edge/max_dc, n_micro/max_dc,
          sum_free_pes/max_hosts] — all in [0, 1]
      4. struct_proj(descriptor) → struct_token [B, 1, D]
      5. Job encoding: job_proj → cross-attention (Q=jobs, K/V=DC slots)
      6. Global encoder over [struct_token || DC slots || job tokens]
         struct_token is NEVER masked; DC and job slots have proper padding masks
      7. Learned pool query → head → features_dim
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

        host_input_dim = dc_type_dim + 1
        self.host_proj = nn.Linear(host_input_dim, hidden_dim)

        host_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.host_encoder = nn.TransformerEncoder(
            host_enc_layer, num_layers=1, enable_nested_tensor=False
        )

        # SWAT-specific: topology descriptor → structure prefix token
        self.struct_proj = nn.Linear(_STRUCT_DIM, hidden_dim)

        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, hidden_dim)

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
        clamped = dc_ids.clamp(0, self.max_datacenters)
        dc_repr = torch.zeros(B, self.max_datacenters + 1, D, device=device)
        counts  = torch.zeros(B, self.max_datacenters + 1, 1, device=device)
        dc_repr.scatter_add_(1, clamped.unsqueeze(-1).expand(-1, -1, D), host_repr)
        counts.scatter_add_(1, clamped.unsqueeze(-1), torch.ones(B, H, 1, device=device))
        dc_repr = dc_repr / counts.clamp(min=1.0)
        return dc_repr[:, 1:, :]

    def _topology_descriptor(
        self, host_feats: torch.Tensor, dc_ids_clamped: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute topology descriptor [B, 5] and dc_mask [B, max_dc].

        All descriptor fields are in [0, 1]:
          0: n_active_dcs / max_datacenters
          1: n_cloud_dcs  / max_datacenters  (dc_type rounds to 1)
          2: n_edge_dcs   / max_datacenters  (dc_type rounds to 2)
          3: n_micro_dcs  / max_datacenters  (dc_type rounds to 3)
          4: sum_free_pes across all hosts / max_hosts
        """
        B = host_feats.shape[0]
        device = host_feats.device
        ones_h = torch.ones(B, self.max_hosts, device=device)

        type_acc = torch.zeros(B, self.max_datacenters + 1, device=device)
        slot_cnt = torch.zeros(B, self.max_datacenters + 1, device=device)
        pes_acc  = torch.zeros(B, self.max_datacenters + 1, device=device)

        type_acc.scatter_add_(1, dc_ids_clamped, host_feats[:, :, self.IDX_DC_TYPE])
        slot_cnt.scatter_add_(1, dc_ids_clamped, ones_h)
        pes_acc.scatter_add_(1, dc_ids_clamped, host_feats[:, :, self.IDX_FREE_PES])

        dc_mask          = slot_cnt[:, 1:] > 0                                      # [B, max_dc]
        dc_type_per_slot = (type_acc / slot_cnt.clamp(min=1.0))[:, 1:].round().long()  # [B, max_dc]

        inv_max_dc = 1.0 / self.max_datacenters
        n_dcs   = dc_mask.float().sum(-1, keepdim=True)                          * inv_max_dc
        n_cloud = ((dc_type_per_slot == 1) & dc_mask).float().sum(-1, keepdim=True) * inv_max_dc
        n_edge  = ((dc_type_per_slot == 2) & dc_mask).float().sum(-1, keepdim=True) * inv_max_dc
        n_micro = ((dc_type_per_slot == 3) & dc_mask).float().sum(-1, keepdim=True) * inv_max_dc
        total_pes = pes_acc[:, 1:].sum(-1, keepdim=True) / (self.max_hosts + 1e-6)

        return torch.cat([n_dcs, n_cloud, n_edge, n_micro, total_pes], dim=-1), dc_mask

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        B = infr.shape[0]

        host_feats = infr.view(B, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats  = jobs.view(B, self.max_jobs,  self.JOB_FEAT_DIM)
        dc_ids = host_feats[:, :, self.IDX_DC_ID].long()

        descriptor, dc_mask = self._topology_descriptor(
            host_feats, dc_ids.clamp(0, self.max_datacenters)
        )
        struct_token = self.struct_proj(descriptor).unsqueeze(1)  # [B, 1, D]

        h = self._embed_hosts(host_feats)
        h = self.host_encoder(h, src_key_padding_mask=(dc_ids == 0))

        dc_repr = self._scatter_to_dc(h, dc_ids)  # [B, max_dc, D]

        j = self.job_proj(job_feats)
        j_att, _ = self.cross_attn(j, dc_repr, dc_repr)
        j = self.cross_norm(j + j_att)

        # struct_token is always unmasked; DC slots and jobs use standard padding masks
        struct_no_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        padding_mask   = torch.cat(
            [struct_no_mask, ~dc_mask, ~(job_feats[:, :, 0] > 0)], dim=1
        )

        combined = torch.cat([struct_token, dc_repr, j], dim=1)
        encoded  = self.global_encoder(combined, src_key_padding_mask=padding_mask)

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded)

        return self.head(pooled.squeeze(1))
