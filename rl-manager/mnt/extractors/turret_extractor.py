import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv, LayerNorm


class TurretGNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        params: dict,
        features_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        leaky_relu_negative_slope: float = 0.2,
    ):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = hidden_dim
        self.params = params
        self.num_layers = num_layers
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.datacenters = params["datacenters"]

        # --- 1. Dimensions & Vocabs ---
        self.max_hosts = params["max_hosts"]
        self.max_jobs = params["max_jobs_waiting"]

        # Vocab Sizes (From config + 1)
        self.vocab_dc_ids = params["max_datacenters"] + 1
        self.vocab_dc_types = params["max_datacenter_types"] + 1
        self.vocab_job_sensitivity = params["max_job_delay_sensitivity_levels"] + 1
        # Embedding Dimensions
        self.emb_dim_id = 4
        self.emb_dim_type = 4
        self.emb_dim_sensitivity = 4

        # --- 2. Calculated Input Dimensions ---

        # Host Feature Vector: <Emb(DC_ID), Emb(Type), Free_PEs>
        self.host_input_dim = self.emb_dim_id + self.emb_dim_type + 1

        # Job Feature Vector: <Cores, Emb(Location), Emb(Sensitivity), Deadline>
        self.job_input_dim = 1 + self.emb_dim_id + self.emb_dim_sensitivity + 1

        # Total dimension of flattened job queue for the MLP
        self.job_queue_flat_dim = self.max_jobs * self.job_input_dim

        self.total_active_hosts = self._calculate_total_hosts()
        self.edge_index = self._build_edge_index()

        # --- 3. Embedding Layers ---
        self.emb_layer_location = nn.Embedding(self.vocab_dc_ids, self.emb_dim_id)
        self.emb_layer_type = nn.Embedding(self.vocab_dc_types, self.emb_dim_type)
        self.emb_layer_sensitivity = nn.Embedding(
            self.vocab_job_sensitivity, self.emb_dim_sensitivity
        )

        # --- 4. Graph Path (Hosts) ---
        self.f_in = nn.Sequential(
            nn.Linear(self.host_input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.LayerNorm(hidden_dim),
        )

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
            )
            self.norm_layers.append(LayerNorm(hidden_dim))

        # Readout (Set Transformer)
        self.readout_encoder = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.readout_decoder = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.readout_seed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # --- 5. Job Path (Context MLP) ---
        self.job_encoder = nn.Sequential(
            nn.Linear(self.job_queue_flat_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

        # --- 6. Fusion ---
        fusion_dim = hidden_dim + hidden_dim
        self.linear_out = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, features_dim),
        )

    def _calculate_total_hosts(self):
        count = 0
        for dc in self.datacenters:
            for host_group in dc["hosts"]:
                count += host_group["amount"]
        return count

    def _build_edge_index(self):
        # (Standard logic kept same)
        sources = []
        targets = []
        dc_to_host_indices = {}
        current_idx = 0
        for dc in self.datacenters:
            dc_hosts = []
            for host_group in dc["hosts"]:
                count = host_group["amount"]
                indices = list(range(current_idx, current_idx + count))
                dc_hosts.extend(indices)
                current_idx += count
            dc_to_host_indices[dc["name"]] = dc_hosts

        def add_edge(u, v):
            sources.append(u)
            targets.append(v)
            sources.append(v)
            targets.append(u)

        for dc_name, host_indices in dc_to_host_indices.items():
            for i in host_indices:
                sources.append(i)
                targets.append(i)
                for j in host_indices:
                    if i != j:
                        add_edge(i, j)
            dc_config = next(d for d in self.datacenters if d["name"] == dc_name)
            connections = dc_config.get("connect_to", [])
            for target_dc_ref in connections:
                target_dc_name = (
                    self.datacenters[target_dc_ref]["name"]
                    if isinstance(target_dc_ref, int)
                    else target_dc_ref
                )
                target_hosts = dc_to_host_indices.get(target_dc_name, [])
                for h_a in host_indices:
                    for h_b in target_hosts:
                        add_edge(h_a, h_b)
        return torch.tensor([sources, targets], dtype=torch.long)

    def _recover_indices(self, flat_one_hot, num_items, vocab_size):
        """
        Helper: Reshapes flattened One-Hot vectors (B, N*V) -> (B, N, V)
        and performs argmax to recover indices (B, N).
        """
        batch_size = flat_one_hot.shape[0]
        # Reshape to [Batch, Num_Items, Vocab_Size]
        reshaped = flat_one_hot.view(batch_size, num_items, vocab_size)
        # Argmax to get the index (0 to Vocab-1)
        return torch.argmax(reshaped, dim=-1)

    def forward(self, observations) -> torch.Tensor:
        if not isinstance(observations, dict):
            raise ValueError("TurretGNNExtractor expects a Dict observation space")

        device = observations["free_host_pes"].device
        batch_size = observations["free_host_pes"].shape[0]

        # ==========================================
        # 1. PROCESS HOSTS (One-Hot -> Indices -> Embeddings)
        # ==========================================

        # Recover Indices from One-Hot Vectors
        # Input: [Batch, 280] -> Output: [Batch, 40] (Indices 0-6)
        h_dc_id_idx = self._recover_indices(
            observations["dc_id"], self.max_hosts, self.vocab_dc_ids
        )

        # Input: [Batch, 160] -> Output: [Batch, 40] (Indices 0-3)
        h_dc_type_idx = self._recover_indices(
            observations["dc_type"], self.max_hosts, self.vocab_dc_types
        )

        h_free_pes = observations["free_host_pes"].float().unsqueeze(-1)

        # Embed Indices
        emb_h_id = self.emb_layer_location(h_dc_id_idx)  # [Batch, 40, 4]
        emb_h_type = self.emb_layer_type(h_dc_type_idx)  # [Batch, 40, 4]

        # Concatenate Node Features
        host_nodes = torch.cat([emb_h_id, emb_h_type, h_free_pes], dim=-1)

        # Slice Active Hosts
        x_active = host_nodes[:, : self.total_active_hosts, :]
        x_flat = x_active.reshape(-1, self.host_input_dim)

        # --- Graph Forward ---
        h = self.f_in(x_flat)
        edge_index = self.edge_index.to(device)
        edge_index_batch = torch.cat(
            [edge_index + i * self.total_active_hosts for i in range(batch_size)], dim=1
        )

        for gat, norm in zip(self.gat_layers, self.norm_layers):
            h_in = h
            h_out = gat(h, edge_index_batch)
            h_out = norm(h_out)
            h = (
                F.leaky_relu(h_out, negative_slope=self.leaky_relu_negative_slope)
                + h_in
            )

        # Readout
        h_dense = h.view(batch_size, self.total_active_hosts, -1)
        h_encoded, _ = self.readout_encoder(h_dense, h_dense, h_dense)
        h_encoded = self.encoder_norm(h_encoded + h_dense)

        seed_expanded = self.readout_seed.expand(batch_size, -1, -1)
        graph_emb, _ = self.readout_decoder(
            query=seed_expanded, key=h_encoded, value=h_encoded
        )
        graph_emb = graph_emb.squeeze(1)

        # ==========================================
        # 2. PROCESS JOBS (One-Hot -> Indices -> Embeddings)
        # ==========================================

        j_cores = observations["job_cores"].float().unsqueeze(-1)
        j_deadline = observations["job_deadline"].float().unsqueeze(-1)

        # Recover Job Indices
        j_loc_idx = self._recover_indices(
            observations["job_location"], self.max_jobs, self.vocab_dc_ids
        )
        j_sens_idx = self._recover_indices(
            observations["job_delay_sensitivity"],
            self.max_jobs,
            self.vocab_job_sensitivity,
        )

        # Embed
        emb_j_loc = self.emb_layer_location(j_loc_idx)
        emb_j_sens = self.emb_layer_sensitivity(j_sens_idx)

        # Concatenate
        job_features = torch.cat([j_cores, emb_j_loc, emb_j_sens, j_deadline], dim=-1)

        # Flatten for MLP
        job_flat = job_features.reshape(batch_size, -1)
        job_emb = self.job_encoder(job_flat)

        # ==========================================
        # 3. FUSION
        # ==========================================
        combined = torch.cat([graph_emb, job_emb], dim=1)

        return self.linear_out(combined)
