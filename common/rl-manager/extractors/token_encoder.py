import math
from typing import NamedTuple

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from extractors.featurize import (
    DC_INPUT_DIM,
    JOB_FEATURES,
    JOB_INPUT_DIM,
    N_DC_TYPES,
    dc_inputs,
    job_inputs,
    masked_mean,
    split_observation,
)


class Tokens(NamedTuple):
    dc: torch.Tensor        # [B, D, H]; slot k is the DC with dc_id k (action k), slot 0 unused
    dc_mask: torch.Tensor   # [B, D] real DCs
    job: torch.Tensor       # [B, J, H]
    job_mask: torch.Tensor  # [B, J] real jobs
    context: torch.Tensor   # [B, 2H] masked means of the DC and job tokens


def _mlp(input_dim: int, dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, dim), nn.ReLU(), nn.LayerNorm(dim),
                         nn.Linear(dim, dim), nn.ReLU())


class MaskedCrossAttention(nn.Module):
    """Queries attend over the keys mask[b, q, k] allows (multi-head, residual, LayerNorm).
    A query with no allowed key receives nothing, so padding rows never produce NaNs."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, n_q, dim = queries.shape
        n_k, d = keys.shape[1], dim // self.heads

        def heads(x, n):
            return x.view(batch, n, self.heads, d).transpose(1, 2)       # [B, h, n, d]

        q, k, v = heads(self.q(queries), n_q), heads(self.k(keys), n_k), heads(self.v(keys), n_k)
        allowed = mask.unsqueeze(1)                                          # [B, 1, Q, K]
        scores = (q @ k.transpose(-1, -2) / math.sqrt(d)).masked_fill(~allowed, -1e9)
        weights = torch.softmax(scores, dim=-1) * allowed
        attended = (weights @ v).transpose(1, 2).reshape(batch, n_q, dim)
        return self.norm(queries + self.out(attended))


class TokenEncoder(BaseFeaturesExtractor):
    """
    Per-DC and per-job tokens for the token heads (extractors/pointer_policy.py).

    A DC token is a shared MLP over the DC's aggregated host features (featurize.dc_inputs),
    a job token a shared MLP over the job's features. With cross_attention, each job then
    attends over the DCs it may use and each DC over the jobs that may use it (masks from
    reach_mask), one residual layer each. Every step is shared across DCs and across jobs and
    every aggregate is masked, so renumbering DCs or reordering job slots permutes the tokens
    and leaves the context unchanged.

    forward() returns the context, [masked mean of DC tokens, masked mean of job tokens].

    Config params (via features_extractor_kwargs):
      token_dim:        token width H (features_dim is 2H)
      cross_attention:  the reach-masked job <-> DC attention layers, the only difference
                        between A5 (on) and A3 SPANE (off), which share the pointer head
    Ablations of A5, each undoing one of its design choices (default off; V1 changes the head
    instead, pointer_policy.PositionalHeadPolicy):
      unmasked_pool:    the DC half of the context is a plain mean over the DC slots, padding
                        included: checklist red flag 3 (slot 0, which holds no DC, stays out,
                        and the job half stays masked) (V2)
      scalar_dc_type:   dc_type enters as one ordinal number instead of one-hot (V3)
      dc_id_embedding:  a learned embedding per DC slot is added to each DC token (V4)
      no_reach:         the cross-attention ignores reach_mask: every real job attends over
                        every real DC and every real DC over every real job (V5)
    """

    def __init__(self, observation_space: spaces.Dict, token_dim: int = 64,
                 cross_attention: bool = True, unmasked_pool: bool = False,
                 scalar_dc_type: bool = False, dc_id_embedding: bool = False,
                 no_reach: bool = False):
        super().__init__(observation_space, features_dim=2 * token_dim)
        n_jobs = observation_space.spaces["jobs_waiting_state"].shape[0] // JOB_FEATURES
        self.n_dc_slots = observation_space.spaces["reach_mask"].shape[0] // n_jobs
        self.token_dim = token_dim
        self.unmasked_pool = unmasked_pool
        self.scalar_dc_type = scalar_dc_type
        self.dc_mlp = _mlp(DC_INPUT_DIM - (N_DC_TYPES - 1 if scalar_dc_type else 0), token_dim)
        self.dc_id_embed = nn.Embedding(self.n_dc_slots, token_dim) if dc_id_embedding else None
        self.job_mlp = _mlp(JOB_INPUT_DIM, token_dim)
        self.cross_attention = cross_attention
        self.no_reach = no_reach
        if cross_attention:
            self.job_to_dc = MaskedCrossAttention(token_dim)
            self.dc_to_job = MaskedCrossAttention(token_dim)

    def tokens(self, observations) -> Tokens:
        device = next(self.parameters()).device
        hosts, jobs, reach = split_observation(observations, device)
        dc_mask, dc_x = dc_inputs(hosts, self.n_dc_slots)
        job_mask, job_x = job_inputs(jobs)
        if self.scalar_dc_type:
            type_id = (dc_x[..., :N_DC_TYPES] * torch.arange(1, N_DC_TYPES + 1, device=device)).sum(-1)
            dc_x = torch.cat([type_id.unsqueeze(-1), dc_x[..., N_DC_TYPES:]], dim=-1)
        dc_tok, job_tok = self.dc_mlp(dc_x), self.job_mlp(job_x)
        if self.dc_id_embed is not None:
            dc_tok = dc_tok + self.dc_id_embed.weight.unsqueeze(0)
        if self.cross_attention:
            usable = dc_mask.unsqueeze(1) & job_mask.unsqueeze(2)                  # [B, J, D]
            if not self.no_reach:
                usable = usable & (reach > 0)
            job_tok, dc_tok = (self.job_to_dc(job_tok, dc_tok, usable),
                               self.dc_to_job(dc_tok, job_tok, usable.transpose(1, 2)))
        if self.unmasked_pool:      # DC slot 0 only collects the padding host rows
            context = torch.cat([dc_tok[:, 1:].mean(dim=1), masked_mean(job_tok, job_mask)], dim=-1)
        else:
            context = torch.cat([masked_mean(dc_tok, dc_mask), masked_mean(job_tok, job_mask)], dim=-1)
        return Tokens(dc_tok, dc_mask, job_tok, job_mask, context)

    def forward(self, observations) -> torch.Tensor:
        return self.tokens(observations).context
