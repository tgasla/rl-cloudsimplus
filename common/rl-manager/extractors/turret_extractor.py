import torch
from torch import nn
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
        # Categorically embed dc_id and dc_type, treat free_vmpes as continuous
        dc_id_dim = min(16, (max_datacenters // 2) + 1)
        dc_type_dim = min(8, (max_dc_types // 2) + 1)
        self.dc_id_embed = nn.Embedding(max_datacenters + 1, dc_id_dim)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)

        host_input_dim = dc_id_dim + dc_type_dim + 1  # +1 for free_vmpes (continuous)
        self.host_proj = nn.Linear(host_input_dim, gnn_hidden)
        self.job_proj = nn.Linear(self.JOB_FEAT_DIM, gnn_hidden)

        # ── Propagation model P ───────────────────────────────────────────────
        # K GATConv layers; each doubles width via multi-head concat
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
        # Set-transformer readout: one learned query attends over all node vectors
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

    def _embed_host_nodes(self, host_feats: torch.Tensor, max_datacenters: int, max_dc_types: int) -> torch.Tensor:
        """
        host_feats: [max_hosts, 3] — (dc_id, dc_type, free_vmpes)
        Returns:    [max_hosts, gnn_hidden]
        """
        dc_ids = host_feats[:, 0].long().clamp(0, max_datacenters)
        dc_types = host_feats[:, 1].long().clamp(0, max_dc_types)
        free_pes = host_feats[:, 2:3].float()
        embedded = torch.cat([
            self.dc_id_embed(dc_ids),
            self.dc_type_embed(dc_types),
            free_pes,
        ], dim=-1)
        return self.host_proj(embedded)

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        infr = observations["infrastructure_state"].float().to(device)
        jobs = observations["jobs_waiting_state"].float().to(device)
        batch_size = infr.shape[0]

        host_feats = infr.view(batch_size, self.max_hosts, self.HOST_FEAT_DIM)
        job_feats = jobs.view(batch_size, self.max_jobs, self.JOB_FEAT_DIM)

        max_dc = self.dc_id_embed.num_embeddings - 1
        max_dct = self.dc_type_embed.num_embeddings - 1

        batch_outs = []
        for b in range(batch_size):
            h = self._embed_host_nodes(host_feats[b], max_dc, max_dct)  # [H, gnn_hidden]
            j = self.job_proj(job_feats[b])                              # [J, gnn_hidden]
            x = torch.cat([h, j], dim=0)                                 # [H+J, gnn_hidden]
            n = x.shape[0]

            # Fully-connected graph (no self-loops)
            idx = torch.arange(n, device=device)
            src, dst = torch.meshgrid(idx, idx, indexing="ij")
            mask = src != dst
            edge_index = torch.stack([src[mask], dst[mask]], dim=0)

            for layer, norm in zip(self.gnn_layers, self.norms):
                x = layer(x, edge_index)
                x = norm(x)
                x = torch.relu(x)

            # Set-transformer readout (learned query attending over all nodes)
            # x: [H+J, out_ch] → unsqueeze to [1, H+J, out_ch]
            x_seq = x.unsqueeze(0)                                         # [1, H+J, D]
            q = self.pool_query                                             # [1, 1, D]
            pooled, _ = self.pool_attn(q, x_seq, x_seq)                   # [1, 1, D]
            batch_outs.append(pooled.squeeze(0).squeeze(0))                # [D]

        out = torch.stack(batch_outs, dim=0)  # [B, out_ch]
        return self.readout(out)
