import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from extractors.featurize import (
    DC_INPUT_DIM,
    JOB_FEATURES,
    JOB_INPUT_DIM,
    dc_inputs,
    job_inputs,
    masked_mean,
    split_observation,
)


def _phi(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class DeepSetsExtractor(BaseFeaturesExtractor):
    """
    A2: DeepSets (Zaheer et al., NeurIPS 2017) over datacenters and over waiting jobs.

    A shared network phi_dc is applied to every DC token (hosts aggregated by dc_id, see
    featurize.dc_inputs) and phi_job to every job; each set is mean-pooled over its real
    elements only; rho maps the two pooled vectors to features_dim. The features are
    invariant to DC order, host order and job order, and do not move with the number of
    padding slots. The action head stays the policy's positional one.

    Config params (via features_extractor_kwargs):
      features_dim: output dimension
      hidden_dim:   width of phi_dc, phi_job and rho
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        n_jobs = observation_space.spaces["jobs_waiting_state"].shape[0] // JOB_FEATURES
        self.n_dc_slots = observation_space.spaces["reach_mask"].shape[0] // n_jobs
        self.phi_dc = _phi(DC_INPUT_DIM, hidden_dim)
        self.phi_job = _phi(JOB_INPUT_DIM, hidden_dim)
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        device = next(self.parameters()).device
        hosts, jobs, _ = split_observation(observations, device)
        dc_mask, dc_x = dc_inputs(hosts, self.n_dc_slots)
        job_mask, job_x = job_inputs(jobs)
        dc_emb = masked_mean(self.phi_dc(dc_x), dc_mask)
        job_emb = masked_mean(self.phi_job(job_x), job_mask)
        return self.rho(torch.cat([dc_emb, job_emb], dim=-1))
