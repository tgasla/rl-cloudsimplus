from functools import partial

import numpy as np
import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from extractors.token_encoder import Tokens


class PointerPolicy(MaskableActorCriticPolicy):
    """
    A5: a pointer head over (job, DC) pairs, on the tokens of a TokenEncoder.

    SB3's head is nn.Linear(features, 32 * 24): a fixed map from job slot and DC slot to logit,
    so renumbering DCs or reordering jobs changes the policy. Here the logit of placing job j
    on DC k is one shared scorer applied to every pair (pointer-network form),

        logit[j, k] = v . tanh(W_job job_j + W_dc dc_k + W_ctx context + w_reach reach_jk),

    the no-op logit is one shared scorer applied to every job, and the critic reads only the
    pooled context. Permuting DCs or job slots therefore permutes the action distribution
    exactly, and the head's cost is linear in jobs + DCs, not in their product.

    reach_input=False drops the w_reach term; with a TokenEncoder without cross-attention
    that is SPANE's advantage module (A3): a shared network scoring each machine from its own
    embedding, the cluster embedding and the request.
    """

    def __init__(self, *args, head_dim: int = 64, reach_input: bool = True, **kwargs):
        self.head_dim = head_dim
        self.reach_input = reach_input
        super().__init__(*args, **kwargs)
        if not self.share_features_extractor:
            raise ValueError("PointerPolicy needs share_features_extractor=True")

    def _build(self, lr_schedule) -> None:
        token_dim = self.features_extractor.token_dim
        self.job_proj = nn.Linear(token_dim, self.head_dim)
        self.dc_proj = nn.Linear(token_dim, self.head_dim, bias=False)
        self.ctx_proj = nn.Linear(2 * token_dim, self.head_dim, bias=False)
        self.reach_proj = nn.Linear(1, self.head_dim, bias=False) if self.reach_input else None
        self.pair_out = nn.Linear(self.head_dim, 1)
        self.noop_head = nn.Sequential(nn.Linear(3 * token_dim, self.head_dim), nn.Tanh(),
                                       nn.Linear(self.head_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * token_dim, self.head_dim), nn.Tanh(),
                                        nn.Linear(self.head_dim, 1))
        if self.ortho_init:
            gains = {self.features_extractor: np.sqrt(2), self.job_proj: np.sqrt(2),
                     self.dc_proj: np.sqrt(2), self.ctx_proj: np.sqrt(2),
                     self.noop_head: np.sqrt(2), self.value_head: np.sqrt(2)}
            if self.reach_proj is not None:
                gains[self.reach_proj] = np.sqrt(2)
            for module, gain in gains.items():
                module.apply(partial(self.init_weights, gain=gain))
            # Small last layers, as SB3 does for action_net (0.01) and value_net (1).
            self.init_weights(self.pair_out, gain=0.01)
            self.init_weights(self.noop_head[-1], gain=0.01)
            self.init_weights(self.value_head[-1], gain=1.0)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict:
        return {**super()._get_constructor_parameters(), "head_dim": self.head_dim,
                "reach_input": self.reach_input}

    def _tokens(self, obs) -> Tokens:
        preprocessed = preprocess_obs(obs, self.observation_space,
                                      normalize_images=self.normalize_images)
        return self.features_extractor.tokens(preprocessed)

    def _logits(self, t: Tokens) -> torch.Tensor:
        """Flat [B, J * D] logits; slot 0 of each job is the no-op (DC slot 0 holds no DC)."""
        pre = (self.job_proj(t.job).unsqueeze(2) + self.dc_proj(t.dc).unsqueeze(1)
               + self.ctx_proj(t.context)[:, None, None, :])
        if self.reach_proj is not None:
            pre = pre + self.reach_proj(t.reach.unsqueeze(-1))
        pair = self.pair_out(torch.tanh(pre)).squeeze(-1)                     # [B, J, D]
        context = t.context.unsqueeze(1).expand(-1, t.job.shape[1], -1)
        noop = self.noop_head(torch.cat([t.job, context], dim=-1))            # [B, J, 1]
        return torch.cat([noop, pair[:, :, 1:]], dim=-1).flatten(1)

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
