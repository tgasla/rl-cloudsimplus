import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from torch_geometric.nn import GATConv
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False

from extractors.token_encoder import Tokens
from extractors.featurize import (
    HOST_INPUT_DIM,
    JOB_INPUT_DIM,
    host_inputs,
    job_inputs,
    split_observation,
)


class TurretGNNExtractor(BaseFeaturesExtractor):
    """
    A4: TURRET's structured policy network (Yang et al., AAAI-24), adapted to job placement.

    TURRET builds its graph from the system's morphology, gives every node its own input
    vector through type-specific input networks (no node identities), propagates with
    multi-head graph attention and reads the graph out with a set transformer. Here:

      nodes   real hosts and real waiting jobs; padding slots stay isolated and are masked
              out of the readout
      edges   host <-> host within the same datacenter, and job <-> host wherever reach_mask
              lets the job use that host's datacenter
      F_in    one MLP for host nodes, one for job nodes (featurize.py inputs)
      P       num_layers GATConv layers with gnn_heads heads (concat), LayerNorm per node
      F_read  one learned seed query attending over the real nodes (set-transformer pooling)

    TURRET's per-node action outputs are extractors.pointer_policy.PerJobPolicy, which reads
    the job nodes' final representations through tokens(). Not included: TURRET's multi-source
    transfer weighting (transfer here is single-source).

    Config params (via features_extractor_kwargs):
      features_dim, gnn_hidden, gnn_heads, num_layers, dropout
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        gnn_hidden: int = 64,
        gnn_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "TurretGNNExtractor requires torch_geometric. "
                "Install with: pip install torch_geometric"
            )
        super().__init__(observation_space, features_dim)

        # ── Input model F_in ─────────────────────────────────────────────────
        self.host_in = nn.Sequential(nn.Linear(HOST_INPUT_DIM, gnn_hidden), nn.ReLU())
        self.job_in = nn.Sequential(nn.Linear(JOB_INPUT_DIM, gnn_hidden), nn.ReLU())

        # ── Propagation model P ───────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = gnn_hidden if i == 0 else gnn_hidden * gnn_heads
            self.gnn_layers.append(
                GATConv(in_ch, gnn_hidden, heads=gnn_heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(gnn_hidden * gnn_heads))
        out_ch = gnn_hidden * gnn_heads
        self.token_dim = out_ch

        # ── Readout model F_read ─────────────────────────────────────────────
        self.pool_query = nn.Parameter(torch.randn(1, 1, out_ch))
        self.pool_attn = nn.MultiheadAttention(
            out_ch, gnn_heads, dropout=dropout, batch_first=True
        )
        self.readout = nn.Sequential(
            nn.Linear(out_ch, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    @staticmethod
    def _edges(dc_ids, host_mask, job_mask, reach) -> torch.Tensor:
        """edge_index over the batch graph; sample b owns nodes [b*N, (b+1)*N), hosts first."""
        batch, n_hosts = dc_ids.shape
        n_jobs = job_mask.shape[1]
        n_nodes = n_hosts + n_jobs

        same_dc = (dc_ids.unsqueeze(2) == dc_ids.unsqueeze(1)) \
            & host_mask.unsqueeze(2) & host_mask.unsqueeze(1)
        same_dc &= ~torch.eye(n_hosts, dtype=torch.bool, device=dc_ids.device)
        b, i, k = same_dc.nonzero(as_tuple=True)
        host_host = torch.stack([b * n_nodes + i, b * n_nodes + k])

        # reach column dc_id says whether the job may use that host's datacenter
        usable = reach.bool().gather(2, dc_ids.unsqueeze(1).expand(batch, n_jobs, n_hosts)) \
            & job_mask.unsqueeze(2) & host_mask.unsqueeze(1)
        b, j, h = usable.nonzero(as_tuple=True)
        job_host = torch.stack([b * n_nodes + n_hosts + j, b * n_nodes + h])
        return torch.cat([host_host, job_host, job_host.flip(0)], dim=1)

    def tokens(self, observations) -> Tokens:
        """Job nodes' final representations and the readout (DC tokens are not formed)."""
        device = next(self.parameters()).device
        hosts, jobs, reach = split_observation(observations, device)
        dc_ids, host_mask, host_x = host_inputs(hosts)
        job_mask, job_x = job_inputs(jobs)
        batch = hosts.shape[0]

        x = torch.cat([self.host_in(host_x), self.job_in(job_x)], dim=1)  # [B, N, hidden]
        n_nodes = x.shape[1]
        edge_index = self._edges(dc_ids, host_mask, job_mask, reach)

        h = x.reshape(batch * n_nodes, -1)
        for layer, norm in zip(self.gnn_layers, self.norms):
            h = torch.relu(norm(layer(h, edge_index)))
        h = h.view(batch, n_nodes, -1)

        # Set-transformer readout over the real nodes: [B, N, D] -> [B, D]
        padding = ~torch.cat([host_mask, job_mask], dim=1)
        padding[padding.all(dim=1)] = False  # an all-padding sample would give NaN weights
        q = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.pool_attn(q, h, h, key_padding_mask=padding)
        n_hosts = host_mask.shape[1]
        return Tokens(dc=None, dc_mask=None, job=h[:, n_hosts:], job_mask=job_mask, reach=reach,
                      context=self.readout(pooled.squeeze(1)))

    def forward(self, observations) -> torch.Tensor:
        return self.tokens(observations).context
