import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.nn import LayerNorm as GeoLayerNorm
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False


class TurretGNNExtractor(BaseFeaturesExtractor):
    """
    GNN-based feature extractor faithful to the TURRET paper (Yang et al., AAAI-24).

    Architecture:
      1. Input model F_in: project each host/job node to gnn_hidden via separate MLPs.
         dc_id and dc_type are embedded categorically; free_vmpes is continuous.
      2. Propagation model P: K GATConv layers with multi-head attention (concat=True),
         applied over a fully-connected host+job graph.
      3. Readout model F_read: set-transformer readout — a learned query vector attends
         over all node representations (attention pooling, equivalent to the paper's
         encoder-decoder readout producing a fixed-dim S_emb).

    Observation format (JobPlacementEnv flat layout):
      infrastructure_state: flat [max_hosts * 3]  — (dc_id, dc_type, free_vmpes) per host
      jobs_waiting_state:   flat [max_jobs * 4]   — (cores, location, sensitivity, deadline)

    Config params (via features_extractor_kwargs):
      features_dim, gnn_hidden, gnn_heads, num_layers, dropout,
      max_datacenters, max_dc_types
    """

    HOST_FEAT_DIM = 3
    JOB_FEAT_DIM = 4

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        gnn_hidden: int = 64,
        gnn_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_datacenters: int = 8,
        max_dc_types: int = 3,
    ):
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "TurretGNNExtractor requires torch_geometric. "
                "Install with: pip install torch_geometric"
            )
        super().__init__(observation_space, features_dim)

        infr_flat = int(np.prod(observation_space.spaces["infrastructure_state"].shape))
        jobs_flat = int(np.prod(observation_space.spaces["jobs_waiting_state"].shape))
        self.max_hosts = infr_flat // self.HOST_FEAT_DIM
        self.max_jobs = jobs_flat // self.JOB_FEAT_DIM

        # ── Input model F_in ─────────────────────────────────────────────────
        dc_id_dim = min(16, (max_datacenters // 2) + 1)
        dc_type_dim = min(8, (max_dc_types // 2) + 1)
        self.dc_id_embed = nn.Embedding(max_datacenters + 1, dc_id_dim)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)

        host_input_dim = dc_id_dim + dc_type_dim + 1  # +1 for free_vmpes (continuous)
        self.host_proj = nn.Linear(host_input_dim, gnn_hidden)
        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, gnn_hidden)

        # ── Propagation model P ───────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = gnn_hidden if i == 0 else gnn_hidden * gnn_heads
            self.gnn_layers.append(
                GATConv(in_ch, gnn_hidden, heads=gnn_heads, dropout=dropout, concat=True)
            )
            self.norms.append(GeoLayerNorm(gnn_hidden * gnn_heads))

        out_ch = gnn_hidden * gnn_heads

        # ── Readout model F_read ─────────────────────────────────────────────
        # Set-transformer readout: one learned query attends over all node vectors.
        # Equivalent to the attention-based ENCODER-DECODER in the TURRET paper.
        self.pool_query = nn.Parameter(torch.randn(1, 1, out_ch))
        self.pool_attn = nn.MultiheadAttention(
            out_ch, gnn_heads, dropout=dropout, batch_first=True
        )
        self.readout = nn.Sequential(
            nn.Linear(out_ch, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

        # ── Precomputed fixed edge_index ─────────────────────────────────────
        # Graph topology (fully-connected, no self-loops) is identical for every
        # sample and every step: n = max_hosts + max_jobs never changes.
        # Registering as a buffer moves it to the correct device automatically.
        n = self.max_hosts + self.max_jobs
        idx = torch.arange(n)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        mask = src != dst
        self.register_buffer("_edge_index", torch.stack([src[mask], dst[mask]], dim=0))

    def _gnn_forward(self, xb: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.gnn_layers, self.norms):
            xb = layer(xb, self._edge_index)
            xb = norm(xb)
            xb = torch.relu(xb)
        return xb

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        batch_size = infr.shape[0]

        host_feats = infr.view(batch_size, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats = jobs.view(batch_size, self.max_jobs, self.JOB_FEAT_DIM)

        max_dc = self.dc_id_embed.num_embeddings - 1
        max_dct = self.dc_type_embed.num_embeddings - 1

        # Vectorized host embedding over full batch — no Python loop
        # [B, H, 3] → [B, H, gnn_hidden]
        dc_ids = host_feats[..., 0].long().clamp(0, max_dc)
        dc_types = host_feats[..., 1].long().clamp(0, max_dct)
        h = self.host_proj(torch.cat([
            self.dc_id_embed(dc_ids),
            self.dc_type_embed(dc_types),
            host_feats[..., 2:3],
        ], dim=-1))

        # [B, J, 4] → [B, J, gnn_hidden]
        j = self.job_proj(job_feats)

        # [B, H+J, gnn_hidden]
        x = torch.cat([h, j], dim=1)

        # GATConv per-sample using precomputed edge_index.
        # Fully-connected graphs are O(n²) edges — batching all B samples at once OOMs.
        # Gradient checkpointing discards GATConv intermediate activations (~140 MB/sample)
        # and recomputes them during backward, reducing peak memory from B×140 MB to ~140 MB.
        node_outs = []
        for b in range(batch_size):
            if self.training:
                xb = grad_checkpoint(self._gnn_forward, x[b], use_reentrant=False)
            else:
                xb = self._gnn_forward(x[b])
            node_outs.append(xb)
        x = torch.stack(node_outs, dim=0)  # [B, n, out_ch]

        # Set-transformer readout: [B, n, D] → pool → [B, D]
        q = self.pool_query.expand(batch_size, -1, -1)   # [B, 1, D]
        pooled, _ = self.pool_attn(q, x, x)              # [B, 1, D]
        return self.readout(pooled.squeeze(1))            # [B, features_dim]
