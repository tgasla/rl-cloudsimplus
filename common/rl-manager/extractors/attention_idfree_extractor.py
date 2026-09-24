import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class IDFreeAttentionExtractor(BaseFeaturesExtractor):
    """
    Identical to AttentionPoolingFeatureExtractor but with dc_id embeddings removed.

    AttentionPoolingFeatureExtractor embeds both dc_id and dc_type per host.
    dc_id embeddings are problematic for cross-environment transfer:
      - Downscale (B→A): DC IDs may be renumbered when a type disappears;
        the embedding for ID 1 (trained on cloud) now gets applied to an edge DC.
      - Upscale (B→C): New DCs in Env C have IDs unseen during training;
        their embeddings are randomly initialised and never trained.

    This extractor uses only [dc_type_emb, free_pes] per host. dc_id is still
    used as a grouping index in _scatter_to_dc (array indexing, not a learned
    association), which is safe across environments.

    Architecture (same as AttentionPoolingFeatureExtractor):
      1. Embed each host: dc_type_emb(dc_type_dim) + free_vmpes → host_proj → hidden_dim
      2. Host self-attention (Transformer encoder, padding mask on inactive hosts)
      3. DC aggregation: scatter-mean by dc_id value → DC tokens [B, max_dc, D]
      4. Job encoding: Linear(4, hidden_dim) per job
      5. Cross-attention: jobs (Q) attend to DC tokens (K/V)
      6. Global Transformer over [DC tokens ‖ job tokens]
      7. Learned pool query attends over all tokens → content-adaptive summary
      8. Linear head → features_dim
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

        self.max_hosts = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs  = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters

        # dc_type ∈ [0, max_dc_types]: 0 = inactive/padding, 1..T = cloud/edge/micro
        # dc_id is NOT embedded — only used as a grouping index in _scatter_to_dc
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
        return self.host_proj(torch.cat([
            self.dc_type_embed(dc_types),
            free_pes,
        ], dim=-1))

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
        return dc_repr[:, 1:, :]  # drop slot 0 (padding accumulator)

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

        dc_repr = self._scatter_to_dc(h, dc_ids)
        j = self.job_proj(job_feats)
        j_att, _ = self.cross_attn(j, dc_repr, dc_repr)
        j = self.cross_norm(j + j_att)

        combined = torch.cat([dc_repr, j], dim=1)
        encoded  = self.global_encoder(combined)

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded)
        pooled = pooled.squeeze(1)

        return self.head(pooled)
