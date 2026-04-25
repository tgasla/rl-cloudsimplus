"""JobPlacementEnv — job-to-datacenter placement RL environment.

Inherits from CloudSimBaseEnv which provides all shared gRPC wiring.
Concrete domain-specific implementation for job placement problem:
- Flat per-host [dc_id, dc_type, free_vmpes] observation
- [dc_index, dc_index, ...] action space (one per waiting job)
- Job placement across multiple datacenters
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .base import CloudSimBaseEnv


class JobPlacementEnv(CloudSimBaseEnv):
    """
    Job placement Gymnasium environment bridging Stable Baselines3 to
    CloudSim Plus via gRPC.

    The agent decides which datacenter to place each waiting job into.
    Observation is flat per-host: [dc_id, dc_type, free_vmpes] per host,
    plus per-job attributes [cores, location, sensitivity, deadline].

    Action space: MultiDiscrete([max_datacenters] * max_jobs_waiting)
        action[i] = DC index to place job i

    Inherits from CloudSimBaseEnv:
        - gRPC client (_client)
        - _sim_id, _rl_problem
        - reset(), step(), close(), ping()
        - _pad_observation()
    """

    DC_TYPE_IDS = {"cloud": 0, "edge": 1, "micro": 2}

    def __init__(
        self,
        params: dict,
        jobs_as_json: str = "[]",
        host: str = "localhost",
        port: int = 50051,
        render_mode: str = None,
    ):
        # Initialize base class (sets up _client, _sim_id=None, _rl_problem=None)
        super().__init__(params, jobs_as_json, host, port, render_mode)

        # Domain-specific RL problem type
        self._rl_problem = "job_placement"

        # ── Domain-specific fields ─────────────────────────────────────────────
        self.max_datacenters = params["max_datacenters"]
        self.max_hosts = params["max_hosts"]
        self.max_jobs_waiting = params["max_jobs_waiting"]
        self.max_pes_per_vm = params.get("max_pes_per_vm", params.get("max_host_pes", 16))
        self.cloudlet_to_dc_assignment_policy = params.get(
            "cloudlet_to_dc_assignment_policy", "rl"
        )

        # ── Observation spaces ─────────────────────────────────────────────────
        # infrastructure_observation: [dc_id-1, dc_type_id, free_vmpes] per host
        # 3 values per host, shape = (3 * total_hosts,)
        total_hosts = params.get("total_hosts", self.max_hosts * self.max_datacenters)
        self.total_hosts = total_hosts
        self.infr_obs_length = 3 * total_hosts
        self.infr_obs_space = spaces.Box(
            low=0,
            high=self.max_pes_per_vm,
            shape=(self.infr_obs_length,),
            dtype=np.int16,
        )

        # jobs_waiting_observation: [cores, location, sensitivity, deadline] per job
        # 4 values per job, shape = (max_jobs_waiting * 4,)
        self.job_obs_length = 4 * self.max_jobs_waiting
        max_val = max(self.max_pes_per_vm, 1000)
        self.job_waiting_obs_space = spaces.Box(
            low=0,
            high=max_val,
            shape=(self.job_obs_length,),
            dtype=np.int16,
        )

        self.observation_space = spaces.Dict(
            {
                "infrastructure_state": self.infr_obs_space,
                "jobs_waiting_state": self.job_waiting_obs_space,
            }
        )

        # ── Action space ──────────────────────────────────────────────────────
        # For RL mode: MultiDiscrete([max_datacenters] * max_jobs_waiting)
        # Only the first N elements (N = number of waiting jobs) are valid
        self.action_space = spaces.MultiDiscrete(
            np.array([self.max_datacenters] * self.max_jobs_waiting)
        )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn("Invalid render mode. Allowed: ['human', 'ansi']")

        # ── Create simulation (CloudSimBaseEnv has _client and _sim_id ready) ─
        import json
        from utils.misc import _params_to_java_format
        java_params = _params_to_java_format(params)
        self._sim_id = self._client.create_simulation(
            json.dumps(java_params), jobs_as_json
        )

    # ── CloudSimBaseEnv abstract methods ───────────────────────────────────────

    def _detect_rl_problem(self) -> str:
        return "job_placement"

    def action_masks(self) -> list[bool]:
        """Return action mask for job placement.

        Returns a flat list of booleans for MultiDiscrete masking.
        Currently allows all DC choices for all jobs.
        """
        masks = []
        for _ in range(self.max_jobs_waiting):
            masks.append([True] * self.max_datacenters)
        return [item for sublist in masks for item in sublist]

    def _get_observation(self, raw_obs: dict) -> dict:
        """Convert raw gRPC observation to job placement gymnasium obs dict."""
        # Infrastructure: [dc_id-1, dc_type_id, free_vmpes] per host
        raw_infr = raw_obs.get("infrastructure_observation", [])
        infr_obs = np.array(raw_infr, dtype=np.int16)
        infr_obs = self._pad_observation(infr_obs, self.infr_obs_length)

        # Jobs waiting: [cores, location, sensitivity, deadline] per job
        raw_jobs = raw_obs.get("jobs_waiting_observation", [])
        jobs_obs = np.array(raw_jobs, dtype=np.int16)
        jobs_obs = self._pad_observation(jobs_obs, self.job_obs_length)

        return {
            "infrastructure_state": infr_obs,
            "jobs_waiting_state": jobs_obs,
        }

    def _parse_step_info(self, raw_info: dict) -> dict:
        """Convert raw gRPC step info to job placement info dict."""
        return {
            "jobs_waiting": raw_info.get("jobs_waiting", 0),
            "jobs_placed": raw_info.get("jobs_placed", 0),
            "jobs_placed_ratio": raw_info.get("jobs_placed_ratio", 0.0),
            "quality_ratio": raw_info.get("quality_ratio", 0.0),
            "deadline_violation_ratio": raw_info.get("deadline_violation_ratio", 0.0),
            "job_wait_time": raw_info.get("job_wait_time", []),
            "is_valid": raw_info.get("is_valid", True),
        }

    def _pad_observation(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad observation array to target dimension."""
        if len(obs) >= target_dim:
            return np.array(obs[:target_dim], dtype=np.int16)
        padded = np.zeros(target_dim, dtype=np.int16)
        padded[: len(obs)] = obs
        return padded