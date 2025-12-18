import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        leaky_relu_negative_slope: float = 0.2,
        use_masking: bool = True,
        use_host_ids: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        self.use_masking = use_masking
        self.use_host_ids = use_host_ids

        # --- 1. Dimensions ---
        self.params = {
            "max_hosts": observation_space["free_host_pes"].shape[0],
            "max_jobs": observation_space["job_cores"].shape[0],
            "vocab_dc_id": observation_space["dc_id"].nvec[0],
            "vocab_dc_type": observation_space["dc_type"].nvec[0],
            "vocab_job_loc": observation_space["job_location"].nvec[0],
            "vocab_job_sens": observation_space["job_delay_sensitivity"].nvec[0],
        }

        # Definition of embedding sizes
        self.emb_dim_id = 8
        self.emb_dim_type = 4

        # --- 2. Host Input Dimension Logic ---
        # If we use IDs, the input is [ID(8) + Type(4) + PEs(1)]
        # If we IGNORE IDs, the input is [Type(4) + PEs(1)]
        if self.use_host_ids:
            self.host_input_dim = self.emb_dim_id + self.emb_dim_type + 1
        else:
            self.host_input_dim = self.emb_dim_type + 1  # Only Type and FreePes

        # --- 3. Embeddings ---
        # We rename this to 'emb_loc' because we still might need it
        # to understand Job Locations, even if we ignore Host IDs.
        self.emb_loc = nn.Embedding(self.params["vocab_dc_id"] + 1, self.emb_dim_id)
        self.emb_dc_type = nn.Embedding(
            self.params["vocab_dc_type"] + 1, self.emb_dim_type
        )

        self.host_encoder = nn.Sequential(
            nn.Linear(self.host_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

        # --- 4. Job Input Logic ---
        # Jobs usually need to know where they are (Location)
        self.job_input_dim = 1 + self.emb_dim_id + 4 + 1
        self.emb_job_sens = nn.Embedding(self.params["vocab_job_sens"] + 1, 4)

        self.job_encoder = nn.Sequential(
            nn.Linear(self.job_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

        # --- 5. Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 6. Output ---
        total_flat_dim = (self.params["max_hosts"] * hidden_dim) + (
            self.params["max_jobs"] * hidden_dim
        )

        self.final_projection = nn.Sequential(
            nn.Linear(total_flat_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

    def _recover_indices(self, flat_one_hot, num_items):
        batch_size = flat_one_hot.shape[0]
        total_features = flat_one_hot.shape[1]
        if total_features % num_items != 0:
            raise ValueError(
                f"Tensor size {total_features} not divisible by items {num_items}"
            )
        vocab_size = total_features // num_items
        reshaped = flat_one_hot.view(batch_size, num_items, vocab_size)
        return torch.argmax(reshaped, dim=-1)

    def forward(self, observations):
        batch_size = observations["free_host_pes"].shape[0]

        # ==========================================
        # 1. PROCESS HOSTS
        # ==========================================

        # We always recover the indices (needed for masking logic anyway)
        h_id_idx = self._recover_indices(
            observations["dc_id"], self.params["max_hosts"]
        )
        h_type_idx = self._recover_indices(
            observations["dc_type"], self.params["max_hosts"]
        )
        h_pes = observations["free_host_pes"].float().unsqueeze(-1)

        # Embed Type
        e_h_type = self.emb_dc_type(h_type_idx)

        # [DECISION POINT] Do we include ID embedding?
        if self.use_host_ids:
            e_h_id = self.emb_loc(h_id_idx)
            # Concatenate ID, Type, PEs
            host_raw = torch.cat([e_h_id, e_h_type, h_pes], dim=-1)
        else:
            # IGNORE ID. Concatenate only Type, PEs
            host_raw = torch.cat([e_h_type, h_pes], dim=-1)

        host_enc = self.host_encoder(host_raw)

        # ==========================================
        # 2. PROCESS JOBS
        # ==========================================
        j_cores = observations["job_cores"].float().unsqueeze(-1)
        j_dead = observations["job_deadline"].float().unsqueeze(-1)

        j_loc_idx = self._recover_indices(
            observations["job_location"], self.params["max_jobs"]
        )
        j_sens_idx = self._recover_indices(
            observations["job_delay_sensitivity"], self.params["max_jobs"]
        )

        # We keep Location for jobs (it might imply 'local' vs 'remote' cost)
        e_j_loc = self.emb_loc(j_loc_idx)
        e_j_sens = self.emb_job_sens(j_sens_idx)

        job_raw = torch.cat([j_cores, e_j_loc, e_j_sens, j_dead], dim=-1)
        job_enc = self.job_encoder(job_raw)

        # ==========================================
        # 3. ATTENTION & MASKING
        # ==========================================
        combined_seq = torch.cat([host_enc, job_enc], dim=1)

        if self.use_masking:
            # Masking logic stays the same (we still use IDs to detect padding 0s)
            host_mask = (h_id_idx == 0) & (h_type_idx == 0) & (h_pes.squeeze(-1) == 0)
            job_mask = (j_cores.squeeze(-1) == 0) & (j_loc_idx == 0)
            combined_mask = torch.cat([host_mask, job_mask], dim=1)

            attended_seq = self.transformer(
                combined_seq, src_key_padding_mask=combined_mask
            )
            mask_expanded = combined_mask.unsqueeze(-1).float()
            attended_seq = attended_seq * (1.0 - mask_expanded)
        else:
            attended_seq = self.transformer(combined_seq)

        flat = attended_seq.reshape(batch_size, -1)
        return self.final_projection(flat)


class AttentionPoolingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        leaky_relu_negative_slope: float = 0.2,
        use_masking: bool = True,
        use_host_ids: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        self.use_masking = use_masking
        self.use_host_ids = use_host_ids

        # --- 1. Dimensions ---
        self.params = {
            "max_hosts": observation_space["free_host_pes"].shape[0],
            "max_jobs": observation_space["job_cores"].shape[0],
            "vocab_dc_id": observation_space["dc_id"].nvec[0],
            "vocab_dc_type": observation_space["dc_type"].nvec[0],
            "vocab_job_loc": observation_space["job_location"].nvec[0],
            "vocab_job_sens": observation_space["job_delay_sensitivity"].nvec[0],
        }

        self.emb_dim_id = 8
        self.emb_dim_type = 4

        # Host Input
        if self.use_host_ids:
            self.host_input_dim = self.emb_dim_id + self.emb_dim_type + 1
        else:
            self.host_input_dim = self.emb_dim_type + 1

        # Embeddings
        self.emb_loc = nn.Embedding(self.params["vocab_dc_id"] + 1, self.emb_dim_id)
        self.emb_dc_type = nn.Embedding(
            self.params["vocab_dc_type"] + 1, self.emb_dim_type
        )

        self.host_encoder = nn.Sequential(
            nn.Linear(self.host_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

        # Job Input
        self.job_input_dim = 1 + self.emb_dim_id + 4 + 1
        self.emb_job_sens = nn.Embedding(self.params["vocab_job_sens"] + 1, 4)

        self.job_encoder = nn.Sequential(
            nn.Linear(self.job_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- NEW OUTPUT LOGIC (POOLING) ---
        # Host Output: We will concat Mean_Pool (Hidden) + Max_Pool (Hidden) = 2 * Hidden
        # Job Output: We flatten jobs because distinct job order matters = Max_Jobs * Hidden
        self.pooled_host_dim = hidden_dim * 2
        self.flat_job_dim = self.params["max_jobs"] * hidden_dim

        total_out_dim = self.pooled_host_dim + self.flat_job_dim

        self.final_projection = nn.Sequential(
            nn.Linear(total_out_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
        )

    def _recover_indices(self, flat_one_hot, num_items):
        batch_size = flat_one_hot.shape[0]
        total_features = flat_one_hot.shape[1]
        if total_features % num_items != 0:
            raise ValueError(
                f"Tensor size {total_features} not divisible by items {num_items}"
            )
        vocab_size = total_features // num_items
        reshaped = flat_one_hot.view(batch_size, num_items, vocab_size)
        return torch.argmax(reshaped, dim=-1)

    def forward(self, observations):
        device = observations["free_host_pes"].device
        batch_size = observations["free_host_pes"].shape[0]

        # --- 1. PROCESS HOSTS ---
        h_id_idx = self._recover_indices(
            observations["dc_id"], self.params["max_hosts"]
        )
        h_type_idx = self._recover_indices(
            observations["dc_type"], self.params["max_hosts"]
        )
        h_pes = observations["free_host_pes"].float().unsqueeze(-1)

        e_h_type = self.emb_dc_type(h_type_idx)

        if self.use_host_ids:
            e_h_id = self.emb_loc(h_id_idx)
            host_raw = torch.cat([e_h_id, e_h_type, h_pes], dim=-1)
        else:
            host_raw = torch.cat([e_h_type, h_pes], dim=-1)

        host_enc = self.host_encoder(host_raw)  # [Batch, N_Hosts, Hidden]

        # --- 2. PROCESS JOBS ---
        j_cores = observations["job_cores"].float().unsqueeze(-1)
        j_dead = observations["job_deadline"].float().unsqueeze(-1)
        j_loc_idx = self._recover_indices(
            observations["job_location"], self.params["max_jobs"]
        )
        j_sens_idx = self._recover_indices(
            observations["job_delay_sensitivity"], self.params["max_jobs"]
        )

        e_j_loc = self.emb_loc(j_loc_idx)
        e_j_sens = self.emb_job_sens(j_sens_idx)

        job_raw = torch.cat([j_cores, e_j_loc, e_j_sens, j_dead], dim=-1)
        job_enc = self.job_encoder(job_raw)  # [Batch, N_Jobs, Hidden]

        # --- 3. ATTENTION & MASKING ---
        combined_seq = torch.cat([host_enc, job_enc], dim=1)

        if self.use_masking:
            host_mask = (h_id_idx == 0) & (h_type_idx == 0) & (h_pes.squeeze(-1) == 0)
            job_mask = (j_cores.squeeze(-1) == 0) & (j_loc_idx == 0)
            combined_mask = torch.cat([host_mask, job_mask], dim=1)

            attended_seq = self.transformer(
                combined_seq, src_key_padding_mask=combined_mask
            )
            mask_expanded = combined_mask.unsqueeze(-1).float()
            attended_seq = attended_seq * (1.0 - mask_expanded)
        else:
            attended_seq = self.transformer(combined_seq)
            # Create a dummy mask for pooling logic below (all False = keep all)
            host_mask = torch.zeros(
                (batch_size, self.params["max_hosts"]), dtype=torch.bool, device=device
            )

        # --- 4. INVARIANT POOLING ---

        # Split back into Hosts and Jobs
        n_hosts = self.params["max_hosts"]
        # [Batch, N_Hosts, Hidden]
        host_out = attended_seq[:, :n_hosts, :]
        # [Batch, N_Jobs, Hidden]
        job_out = attended_seq[:, n_hosts:, :]

        # === MEAN POOLING ===
        # We must ignore masked (padded) hosts in the average
        if self.use_masking:
            # host_mask is True for padding. Invert it for valid items.
            valid_mask = ~host_mask  # [Batch, N_Hosts]
            valid_mask_float = valid_mask.float().unsqueeze(-1)  # [Batch, N_Hosts, 1]

            # Sum valid vectors
            sum_hosts = (host_out * valid_mask_float).sum(dim=1)
            # Count valid vectors (avoid div by zero)
            count_hosts = valid_mask_float.sum(dim=1).clamp(min=1.0)

            host_mean = sum_hosts / count_hosts

            # === MAX POOLING ===
            # Fill padding with very small number so they aren't picked as max
            host_out_masked = host_out.clone()
            host_out_masked[host_mask] = -1e9
            host_max = host_out_masked.max(dim=1)[0]
        else:
            host_mean = host_out.mean(dim=1)
            host_max = host_out.max(dim=1)[0]

        # Combine Pooled Hosts + Flattened Jobs
        # [Batch, Hidden] + [Batch, Hidden] + [Batch, N_Jobs*Hidden]
        # Result is size-invariant regarding the host count!
        flat_jobs = job_out.reshape(batch_size, -1)

        features = torch.cat([host_mean, host_max, flat_jobs], dim=1)

        return self.final_projection(features)
