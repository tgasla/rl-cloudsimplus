from functools import partial

import numpy as np
import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from extractors.token_encoder import Tokens


class TokenHeadPolicy(MaskableActorCriticPolicy):
    """Shared plumbing for heads that act on an encoder's tokens instead of on SB3's pooled
    features: the encoder provides tokens(obs) (extractors.token_encoder.Tokens), a subclass
    builds its head in _build_head and maps tokens to flat [B, J * D] logits in _logits, and the
    critic reads the pooled context."""

    def __init__(self, *args, head_dim: int = 64, **kwargs):
        self.head_dim = head_dim
        super().__init__(*args, **kwargs)
        if not self.share_features_extractor:
            raise ValueError(f"{type(self).__name__} needs share_features_extractor=True")

    def _build_head(self) -> dict:
        """Create the head modules; return {module: orthogonal-init gain}."""
        raise NotImplementedError

    def _logits(self, t: Tokens) -> torch.Tensor:
        raise NotImplementedError

    def _build(self, lr_schedule) -> None:
        gains = self._build_head()
        self.value_head = nn.Sequential(nn.Linear(self.features_extractor.features_dim, self.head_dim),
                                        nn.Tanh(), nn.Linear(self.head_dim, 1))
        if self.ortho_init:
            gains = {self.features_extractor: np.sqrt(2), self.value_head: np.sqrt(2), **gains}
            for module, gain in gains.items():
                module.apply(partial(self.init_weights, gain=gain))
            self.init_weights(self.value_head[-1], gain=1.0)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict:
        return {**super()._get_constructor_parameters(), "head_dim": self.head_dim}

    def _tokens(self, obs) -> Tokens:
        preprocessed = preprocess_obs(obs, self.observation_space,
                                      normalize_images=self.normalize_images)
        return self.features_extractor.tokens(preprocessed)

    def _distribution(self, t: Tokens, action_masks):
        distribution = self.action_dist.proba_distribution(action_logits=self._logits(t))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        t = self._tokens(obs)
        distribution = self._distribution(t, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, self.value_head(t.context), log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        t = self._tokens(obs)
        distribution = self._distribution(t, action_masks)
        return self.value_head(t.context), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, action_masks=None):
        return self._distribution(self._tokens(obs), action_masks)

    def predict_values(self, obs) -> torch.Tensor:
        return self.value_head(self._tokens(obs).context)


class PointerPolicy(TokenHeadPolicy):
    """
    The pointer head over (job, DC) pairs, on the tokens of a TokenEncoder. A3 (SPANE) and A5
    share it unchanged and differ only in the encoder: A5 = A3 + the reach-masked job <-> DC
    cross-attention. On an encoder without cross-attention it is SPANE's advantage module: a
    shared network scoring each machine from its own embedding, the cluster embedding and the
    request.

    SB3's head is nn.Linear(features, J * D): a fixed map from job slot and DC slot to logit,
    so renumbering DCs or reordering jobs changes the policy. Here the logit of placing job j
    on DC k is one shared scorer applied to every pair (pointer-network form),

        logit[j, k] = v . tanh(W_job job_j + W_dc dc_k + W_ctx context),

    the no-op logit is one shared scorer applied to every job, and the critic reads only the
    pooled context. Permuting DCs or job slots therefore permutes the action distribution
    exactly, and the head's parameter count does not depend on J or D. Its compute does:
    scoring every pair costs O(J * D * head_dim), the same order as the positional head.

    Reach is not a head input. The action mask keeps only pairs with reach 1, so a reach term
    would add the same constant to every legal pair's score.
    """

    def _build_head(self) -> dict:
        token_dim = self.features_extractor.token_dim
        self.job_proj = nn.Linear(token_dim, self.head_dim)
        self.dc_proj = nn.Linear(token_dim, self.head_dim, bias=False)
        self.ctx_proj = nn.Linear(self.features_extractor.features_dim, self.head_dim, bias=False)
        self.pair_out = nn.Linear(self.head_dim, 1)
        self.noop_head = nn.Sequential(
            nn.Linear(token_dim + self.features_extractor.features_dim, self.head_dim),
            nn.Tanh(), nn.Linear(self.head_dim, 1))
        # Small last layers, as SB3 does for action_net (0.01).
        return {self.job_proj: np.sqrt(2), self.dc_proj: np.sqrt(2), self.ctx_proj: np.sqrt(2),
                self.noop_head: np.sqrt(2), self.pair_out: 0.01, self.noop_head[-1]: 0.01}

    def _logits(self, t: Tokens) -> torch.Tensor:
        """Flat [B, J * D] logits; slot 0 of each job is the no-op (DC slot 0 holds no DC)."""
        pre = (self.job_proj(t.job).unsqueeze(2) + self.dc_proj(t.dc).unsqueeze(1)
               + self.ctx_proj(t.context)[:, None, None, :])
        pair = self.pair_out(torch.tanh(pre)).squeeze(-1)                     # [B, J, D]
        context = t.context.unsqueeze(1).expand(-1, t.job.shape[1], -1)
        noop = self.noop_head(torch.cat([t.job, context], dim=-1))            # [B, J, 1]
        return torch.cat([noop, pair[:, :, 1:]], dim=-1).flatten(1)


class PositionalHeadPolicy(TokenHeadPolicy):
    """
    V1: A5's tokens under a positional head. The job tokens, the DC tokens and the context are
    flattened and mapped by an MLP with one hidden layer of head_dim units to the J * D logits,
    so every (job slot, DC slot) pair has its own output weights, as in SB3's head, and
    renumbering DCs or reordering job slots changes the policy. The head reads what the pointer
    head reads, per slot, and the critic is TokenHeadPolicy's: V1 differs from A5 only in the
    head's symmetry. Padding tokens are zeroed first; in the pointer head they only reach
    logits the action mask removes.
    """

    def _build_head(self) -> dict:
        encoder = self.features_extractor
        n_jobs, n_actions = len(self.action_space.nvec), int(self.action_space.nvec[0])
        flat = (n_jobs + encoder.n_dc_slots) * encoder.token_dim + encoder.features_dim
        self.flat_head = nn.Sequential(nn.Linear(flat, self.head_dim), nn.Tanh(),
                                       nn.Linear(self.head_dim, n_jobs * n_actions))
        return {self.flat_head: np.sqrt(2), self.flat_head[-1]: 0.01}

    def _logits(self, t: Tokens) -> torch.Tensor:
        job = t.job * t.job_mask.unsqueeze(-1)
        dc = t.dc * t.dc_mask.unsqueeze(-1)
        return self.flat_head(torch.cat([job.flatten(1), dc.flatten(1), t.context], dim=-1))
