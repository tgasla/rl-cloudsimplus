import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Hierarchical Transformer extractor designed for DC-level job placement decisions.

    The architecture mirrors the action space: since the agent picks a DC index,
    the extractor reasons explicitly at DC granularity, not host granularity.

    Pipeline:
      1. Embed each host: categorical embedding of dc_id + dc_type, continuous free_vmpes.
      2. Host self-attention: context-aware host representations (masked for padding).
      3. DC aggregation: scatter-mean host representations to DC-level tokens.
      4. Job encoding: project [cores, location, sensitivity, deadline] → D.
      5. Cross-attention: jobs (Q) attend to DC tokens (K/V) — jobs learn which DC fits.
      6. Global Transformer encoder over [DC tokens ‖ updated job tokens].
      7. Mean pooling → final embedding.

    Observation format (JobPlacementEnv flat layout):
      infrastructure_state: flat [max_hosts * 3]  — (dc_id, dc_type, free_vmpes) per host
      jobs_waiting_state:   flat [max_jobs * 4]   — (cores, location, sensitivity, deadline)

    Config params (via features_extractor_kwargs):
      features_dim, hidden_dim, n_heads, n_layers, dropout, max_datacenters, max_dc_types
    """

    # Index within each host's 3-tuple
    IDX_DC_ID = 0
    IDX_DC_TYPE = 1
    IDX_FREE_PES = 2

    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM = 4

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
        self.max_jobs = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters

        # ── Host embedding ────────────────────────────────────────────────────
        # dc_id ∈ [0, max_datacenters]: 0 = no-op/padding, 1..N = real DCs
        # dc_type ∈ [0, max_dc_types]: 0 = padding, 1..T = cloud/edge/micro
        dc_id_dim = min(16, (max_datacenters // 2) + 1)
        dc_type_dim = min(8, (max_dc_types // 2) + 1)

        self.dc_id_embed = nn.Embedding(max_datacenters + 1, dc_id_dim)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)

        host_input_dim = dc_id_dim + dc_type_dim + 1  # +1 continuous for free_vmpes
        self.host_proj = nn.Linear(host_input_dim, hidden_dim)

        # ── Step 2: Host self-attention ───────────────────────────────────────
        host_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.host_encoder = nn.TransformerEncoder(host_enc_layer, num_layers=1, enable_nested_tensor=False)

        # ── Step 4: Job encoding ──────────────────────────────────────────────
        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, hidden_dim)

        # ── Step 5: Cross-attention (jobs → DCs) ─────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ── Step 6: Global Transformer over [DC ‖ job] tokens ────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        # ── Step 7: Mean pooling → output ─────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
        )

    def _embed_hosts(self, host_feats: torch.Tensor) -> torch.Tensor:
        """
        host_feats: [B, max_hosts, 3]
        Returns:    [B, max_hosts, hidden_dim]
        """
        dc_ids = host_feats[:, :, self.IDX_DC_ID].long().clamp(0, self.max_datacenters)
        dc_types = host_feats[:, :, self.IDX_DC_TYPE].long().clamp(0, self.dc_type_embed.num_embeddings - 1)
        free_pes = host_feats[:, :, self.IDX_FREE_PES].unsqueeze(-1)
        return self.host_proj(torch.cat([
            self.dc_id_embed(dc_ids),
            self.dc_type_embed(dc_types),
            free_pes,
        ], dim=-1))

    def _scatter_to_dc(self, host_repr: torch.Tensor, dc_ids: torch.Tensor) -> torch.Tensor:
        """
        Aggregate host representations into DC-level tokens via scatter-mean.
        Padding hosts (dc_id == 0) are excluded.

        host_repr: [B, max_hosts, D]
        dc_ids:    [B, max_hosts]  — 0 = padding, 1..N = real DC indices

        Returns:   [B, max_datacenters, D]  (index i-1 = DC with dc_id=i)
        """
        B, H, D = host_repr.shape
        device = host_repr.device

        # Allocate [B, max_datacenters+1, D]; slot 0 = padding accumulator (discarded)
        dc_repr = torch.zeros(B, self.max_datacenters + 1, D, device=device)
        counts = torch.zeros(B, self.max_datacenters + 1, 1, device=device)

        idx = dc_ids.clamp(0, self.max_datacenters).unsqueeze(-1).expand(-1, -1, D)
        dc_repr.scatter_add_(1, idx, host_repr)
        counts.scatter_add_(1, dc_ids.clamp(0, self.max_datacenters).unsqueeze(-1),
                            torch.ones(B, H, 1, device=device))
        dc_repr = dc_repr / counts.clamp(min=1.0)

        # Drop slot 0 (padding); return real DC slots 1..max_datacenters
        return dc_repr[:, 1:, :]  # [B, max_datacenters, D]

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        batch_size = infr.shape[0]

        host_feats = infr.view(batch_size, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats = jobs.view(batch_size, self.max_jobs, self.JOB_FEAT_DIM)

        dc_ids = host_feats[:, :, self.IDX_DC_ID].long()

        # Step 1-2: embed + self-attend hosts (mask padding tokens)
        h = self._embed_hosts(host_feats)                          # [B, H, D]
        padding_mask = (dc_ids == 0)                               # True = padding host
        h = self.host_encoder(h, src_key_padding_mask=padding_mask)

        # Step 3: DC-level aggregation
        dc_repr = self._scatter_to_dc(h, dc_ids)                   # [B, max_dc, D]

        # Step 4: job encoding
        j = self.job_proj(job_feats)                               # [B, J, D]

        # Step 5: cross-attention (jobs query DCs)
        j_att, _ = self.cross_attn(j, dc_repr, dc_repr)
        j = self.cross_norm(j + j_att)

        # Step 6: global encoding over DC + job tokens
        combined = torch.cat([dc_repr, j], dim=1)                  # [B, max_dc+J, D]
        encoded = self.global_encoder(combined)

        # Step 7: mean pool → output
        pooled = encoded.mean(dim=1)                               # [B, D]
        return self.head(pooled)


class AttentionPoolingFeatureExtractor(BaseFeaturesExtractor):
    """
    Variant of AttentionFeatureExtractor that replaces mean pooling with a
    learned attention pooling query.

    Same pipeline as AttentionFeatureExtractor through step 6; step 7 uses a
    single learnable query that attends over all encoded tokens, producing a
    content-adaptive summary instead of a uniform average.

    Same config params as AttentionFeatureExtractor.
    """

    IDX_DC_ID = 0
    IDX_DC_TYPE = 1
    IDX_FREE_PES = 2
    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM = 4

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
        self.max_jobs = jobs_flat // self.JOB_FEAT_DIM
        self.max_datacenters = max_datacenters

        dc_id_dim = min(16, (max_datacenters // 2) + 1)
        dc_type_dim = min(8, (max_dc_types // 2) + 1)

        self.dc_id_embed = nn.Embedding(max_datacenters + 1, dc_id_dim)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)

        host_input_dim = dc_id_dim + dc_type_dim + 1
        self.host_proj = nn.Linear(host_input_dim, hidden_dim)

        host_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.host_encoder = nn.TransformerEncoder(host_enc_layer, num_layers=1, enable_nested_tensor=False)

        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        # Learned pooling query (the key difference from AttentionFeatureExtractor)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.head = nn.Linear(hidden_dim, features_dim)

    def _embed_hosts(self, host_feats: torch.Tensor) -> torch.Tensor:
        dc_ids = host_feats[:, :, self.IDX_DC_ID].long().clamp(0, self.max_datacenters)
        dc_types = host_feats[:, :, self.IDX_DC_TYPE].long().clamp(0, self.dc_type_embed.num_embeddings - 1)
        free_pes = host_feats[:, :, self.IDX_FREE_PES].unsqueeze(-1)
        return self.host_proj(torch.cat([
            self.dc_id_embed(dc_ids),
            self.dc_type_embed(dc_types),
            free_pes,
        ], dim=-1))

    def _scatter_to_dc(self, host_repr: torch.Tensor, dc_ids: torch.Tensor) -> torch.Tensor:
        B, H, D = host_repr.shape
        device = host_repr.device
        dc_repr = torch.zeros(B, self.max_datacenters + 1, D, device=device)
        counts = torch.zeros(B, self.max_datacenters + 1, 1, device=device)
        idx = dc_ids.clamp(0, self.max_datacenters).unsqueeze(-1).expand(-1, -1, D)
        dc_repr.scatter_add_(1, idx, host_repr)
        counts.scatter_add_(1, dc_ids.clamp(0, self.max_datacenters).unsqueeze(-1),
                            torch.ones(B, H, 1, device=device))
        dc_repr = dc_repr / counts.clamp(min=1.0)
        return dc_repr[:, 1:, :]

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        batch_size = infr.shape[0]

        host_feats = infr.view(batch_size, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats = jobs.view(batch_size, self.max_jobs, self.JOB_FEAT_DIM)
        dc_ids = host_feats[:, :, self.IDX_DC_ID].long()

        h = self._embed_hosts(host_feats)
        h = self.host_encoder(h, src_key_padding_mask=(dc_ids == 0))

        dc_repr = self._scatter_to_dc(h, dc_ids)
        j = self.job_proj(job_feats)
        j_att, _ = self.cross_attn(j, dc_repr, dc_repr)
        j = self.cross_norm(j + j_att)

        combined = torch.cat([dc_repr, j], dim=1)
        encoded = self.global_encoder(combined)

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded)
        pooled = pooled.squeeze(1)

        return self.head(pooled)
