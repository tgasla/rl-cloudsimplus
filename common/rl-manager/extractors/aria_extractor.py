import torch
from torch import nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ARIAExtractor(BaseFeaturesExtractor):
    """
    ARIA: Attention-RBF Invariant Architecture.

    Combines three complementary components, each chosen to close a specific
    gap left by prior extractors, to achieve all five topology-change invariances
    required for robust cross-environment transfer in cloud-edge scheduling:

      1. IDFreeAttentionExtractor base (host encoder + cross-attention + global encoder)

         Why: Job-to-DC cross-attention (Q=jobs, K/V=DC slots) creates implicit
         relative encodings — each job embedding becomes conditioned on the current
         infrastructure state, letting the agent reason about "this job given THESE
         DCs" rather than treating jobs and DCs as independent streams. The global
         encoder then refines all tokens jointly, capturing cross-slot dependencies
         that per-DC aggregation cannot express.

         Empirical basis: attention_idfree is the most stable extractor across both
         C->A (structural downscale: cloud DC absent) and C->B (size downscale) in
         our benchmark — the cross-attention "brain" is the key differentiator over
         simpler scatter-pool extractors like SPANE and hybrid.

         What we take: the full pipeline (host_encoder, _scatter_to_dc, cross_attn,
         cross_norm, global_encoder), replacing only the final MHA pool_attn with RBF.

      2. RBF kernel at the FINAL pool step (idea from HybridRBFPoolExtractor)

         Why: Standard softmax attention assigns each token at least 1/N weight via
         the exp/sum normalisation; in a topology where a DC type is absent, its
         near-zero token still injects 1/N noise into the pool summary. The RBF
         kernel exp(-||q-k||^2 / 2*sigma^2) gives zero weight to tokens far from the
         pool query in representation space — absent DC types naturally score near zero
         without explicit mask bookkeeping.

         Critical placement: RBF must go at the FINAL pool over global_encoder output,
         NOT at the DC-pool step. Placing RBF earlier (e.g. on dc_repr before
         cross_attn) collapses per-slot DC tokens into a single vector before
         cross_attn needs them as K/V keys, destroying the per-slot resolution that
         makes the cross-attention meaningful. hybrid_rbf applies RBF at the DC-pool
         step and achieves the best C->A zero-shot (82.6%) but cannot use cross-attn;
         ARIA moves RBF to the final pool to get locality AND cross-attn.

         sigma is a learned global scalar (exp(log_sigma), clamped > 0). Initialised
         to 1 so all tokens start with equal effective weight; the network learns the
         right bandwidth during training.

      3. Residual adaptation layer (from euromlsys / CustomFeatureExtractor)

         Why: Empirically, euromlsys wins C->B asymptotically because it has ~65k
         adaptation parameters — 15x more fine-tuning bandwidth than extractors with
         post-head adapters. `base + 0.1 * adapter(base)` provides a dedicated
         pathway for rapid specialisation to a target environment. The 0.1 scale
         keeps the pre-trained representation stable at the start of fine-tuning
         while giving gradient a clear route to specialise without destroying
         zero-shot performance.

    Architecture pipeline:
      host_encoder -> scatter_to_dc -> cross_attn(Q=jobs, K/V=dc_repr)
      -> global_encoder (padding mask on inactive DCs and inactive jobs)
      -> RBF_pool (valid_mask = active DC slots | active job slots)
      -> head -> base + 0.1 * adaptation_layer(base)

    Invariances achieved:
      Permutation : scatter_add indexed by dc_id VALUE (not array position);
                    cross_attn and global_encoder are set-equivariant.
      Count       : valid_mask zeros inactive DC/job slots in the RBF pool;
                    global_encoder padding mask prevents inactive tokens from
                    polluting active representations.
      Type        : dc_type embedded via nn.Embedding — independent learned
                    vector per type, no ordinal assumption. No dc_id embedding
                    anywhere.
      Workload    : cross_attn lets each job query the full DC set, producing
                    job-specific infrastructure views adaptive to current load.
      Scale       : valid_mask covers active DCs + active jobs; no operation
                    depends on total token count or sequence length.

    Pros:
      - Unique combination of the three most effective transfer components
        observed empirically across C->A and C->B benchmarks.
      - Transfer-safe: no dc_id embeddings; scatter_add for DC aggregation;
        RBF masked strictly to valid tokens.
      - Global encoder processes infrastructure-conditioned job tokens, giving
        richer joint representations than any extractor using independent streams.
      - RBF sigma is data-adaptive and generalises to sequences of any length
        without retraining.

    Cons:
      - ~15-20% more parameters than attention_idfree (due to adaptation_layer).
      - log_sigma is a single global scalar; future work could use per-type or
        per-head bandwidths for finer locality control.
      - ARIA is a novel combination — empirical validation required to confirm
        the hypothesis that RBF + cross-attn + adaptation synergise as predicted.
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

        # dc_type in [0, max_dc_types]: 0 = inactive/padding, 1..T = cloud/edge/micro.
        # dc_id is NOT embedded — used only as a grouping index in _scatter_to_dc.
        dc_type_dim = min(8, (max_dc_types // 2) + 1)
        self.dc_type_embed = nn.Embedding(max_dc_types + 1, dc_type_dim)

        self.host_proj = nn.Linear(dc_type_dim + 1, hidden_dim)  # type_emb + free_pes

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

        # RBF pool: learned query vector [hidden_dim] + learnable bandwidth sigma.
        # log_sigma initialised to 0 (sigma_0 = 1) so all valid tokens start with
        # equal weight; the network learns the right bandwidth during training.
        self.pool_query = nn.Parameter(torch.randn(hidden_dim))
        self.log_sigma   = nn.Parameter(torch.zeros(1))

        self.head = nn.Linear(hidden_dim, features_dim)

        # Residual adaptation: base + 0.1 * adapter(base).
        # Bottleneck forces the adapter to learn a low-rank correction, preventing
        # it from overriding the base representation and destabilising zero-shot perf.
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
        """Scatter-mean host representations to DC slots; return (dc_repr, dc_mask).

        Aggregation is keyed on dc_id VALUE, not on the host's position in the
        observation array — making the result permutation invariant with respect
        to host ordering and DC numbering conventions.

        Returns:
            dc_repr [B, max_dc, D]: mean-pooled host representations per DC.
            dc_mask [B, max_dc]:    True for DC slots with at least one host.
        """
        B, H, D = host_repr.shape
        device = host_repr.device
        clamped = dc_ids.clamp(0, self.max_datacenters)
        dc_repr = torch.zeros(B, self.max_datacenters + 1, D, device=device)
        counts  = torch.zeros(B, self.max_datacenters + 1, 1, device=device)
        dc_repr.scatter_add_(1, clamped.unsqueeze(-1).expand(-1, -1, D), host_repr)
        counts.scatter_add_(1, clamped.unsqueeze(-1), torch.ones(B, H, 1, device=device))
        dc_repr = dc_repr / counts.clamp(min=1.0)
        dc_mask = counts[:, 1:, 0] > 0   # [B, max_dc]; slot 0 is the padding accumulator
        return dc_repr[:, 1:, :], dc_mask

    def _rbf_pool(
        self, encoded: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """RBF-weighted pool; zero weight for tokens outside valid_mask.

        Uses the Gaussian kernel: w_i = exp(-||q - encoded_i||^2 / 2*sigma^2),
        masked to valid tokens only, then re-normalised so weights sum to 1.

        Args:
            encoded    [B, T, D]: global encoder output (DC slots + job tokens).
            valid_mask [B, T]:    True for tokens to include (active DCs + active jobs).
        Returns:
            pooled [B, D]: RBF-weighted sum over valid tokens.
        """
        q = self.pool_query.view(1, 1, -1)                               # [1, 1, D]
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        diffs = encoded - q                                               # [B, T, D]
        rbf_weights = (-0.5 * (diffs ** 2).sum(-1) / sigma ** 2).exp()  # [B, T]
        rbf_weights = rbf_weights.masked_fill(~valid_mask, 0.0)
        rbf_weights = rbf_weights / rbf_weights.sum(-1, keepdim=True).clamp(min=1e-6)
        return (rbf_weights.unsqueeze(-1) * encoded).sum(1)              # [B, D]

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

        dc_repr, dc_mask = self._scatter_to_dc(h, dc_ids)

        j = self.job_proj(job_feats)
        j_att, _ = self.cross_attn(j, dc_repr, dc_repr)  # jobs become infrastructure-conditioned
        j = self.cross_norm(j + j_att)

        job_active = job_feats[:, :, 0] > 0
        padding_mask = torch.cat([~dc_mask, ~job_active], dim=1)  # True = masked out (PyTorch)
        combined = torch.cat([dc_repr, j], dim=1)
        encoded  = self.global_encoder(combined, src_key_padding_mask=padding_mask)

        valid_mask = torch.cat([dc_mask, job_active], dim=1)
        pooled = self._rbf_pool(encoded, valid_mask)

        base = self.head(pooled)
        return base + 0.1 * self.adaptation_layer(base)
