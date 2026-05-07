"""JobPlacementEnv — job-to-datacenter placement RL environment.

Inherits from CloudSimBaseEnv which provides all shared gRPC wiring.
Concrete domain-specific implementation for job placement problem:
- Flat per-host [dc_id, dc_type, free_vmpes] observation
- [dc_index, dc_index, ...] action space (one per waiting job)
- Job placement across multiple datacenters
"""

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
    JOB_OBS_FEATURES = 4  # cores, location, delaySensitivity, deadline — must match Java CloudSimProxy.JOB_OBS_FEATURES

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
        self.max_pes_per_vm = params.get("max_pes_per_vm", params.get("max_host_pes"))
        self.cloudlet_to_dc_mapping = params.get("cloudlet_to_dc_mapping", "rl")

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
        self.job_obs_length = self.JOB_OBS_FEATURES * self.max_jobs_waiting
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

        # ── Last observation cache (used by action_masks) ─────────────────────
        self._last_infr_obs = np.zeros(self.infr_obs_length, dtype=np.int16)
        self._last_jobs_obs = np.zeros(self.job_obs_length, dtype=np.int16)

        # ── Create simulation (CloudSimBaseEnv has _client and _sim_id ready) ─
        import json
        self._sim_id = self._client.create_simulation(
            json.dumps(params), jobs_as_json
        )

    # ── CloudSimBaseEnv abstract methods ───────────────────────────────────────

    def action_masks(self) -> list[bool]:
        """Return action mask for MaskablePPO.

        For each (job, dc) pair: valid if DC has free capacity >= job's requested cores.
        Action 0 is always valid (no-op: skip placing this job).
        Padding job slots (cores=0) allow all actions.
        If no real DC can fit a job, fall back to allowing all actions so MaskablePPO
        always has at least one valid choice per sub-space.

        obs_dc_id = cloudSim_dc_id - 1, which equals the agent action for that DC.
        action=0 is no-op; real DCs start at obs_dc_id=1 (action=1).
        """
        infr_obs = self._last_infr_obs
        jobs_obs = self._last_jobs_obs

        # Scatter-max: compute max free PEs per DC action index in one vectorized pass.
        # infr_obs layout: [obs_dc_id, dc_type, free_vmpes] per host.
        n_hosts = self.infr_obs_length // 3
        hosts = infr_obs.reshape(n_hosts, 3)
        dc_ids = hosts[:, 0].astype(np.int64)
        free_pes = hosts[:, 2].astype(np.int64)
        valid = (dc_ids > 0) & (dc_ids < self.max_datacenters)
        dc_max_free = np.zeros(self.max_datacenters, dtype=np.int64)
        np.maximum.at(dc_max_free, dc_ids[valid], free_pes[valid])

        # Broadcast [max_jobs, 1] cores against [1, max_datacenters] capacity.
        cores = jobs_obs.reshape(self.max_jobs_waiting, self.JOB_OBS_FEATURES)[:, 0].astype(np.int64)
        mask_matrix = dc_max_free[np.newaxis, :] >= cores[:, np.newaxis]  # [J, DC]
        mask_matrix[:, 0] = True                        # action=0 (no-op) always valid
        mask_matrix[cores == 0] = True                  # padding slots: allow all
        all_blocked = (cores > 0) & ~mask_matrix[:, 1:].any(axis=1)
        mask_matrix[all_blocked] = True                 # MaskablePPO invariant: ≥1 valid action

        return mask_matrix.ravel().tolist()

    def _get_observation(self, raw_obs: dict) -> dict:
        """Convert raw gRPC observation to job placement gymnasium obs dict."""
        # Infrastructure: [dc_id-1, dc_type_id, free_vmpes] per host
        infr_obs = np.array(raw_obs.get("infrastructure_observation"), dtype=np.int16)
        infr_obs = self._pad_observation(infr_obs, self.infr_obs_length)
        self._last_infr_obs = infr_obs

        # Jobs waiting: [cores, location, sensitivity, deadline] per job
        jobs_obs = np.array(raw_obs.get("secondary_observation"), dtype=np.int16)
        jobs_obs = self._pad_observation(jobs_obs, self.job_obs_length)
        self._last_jobs_obs = jobs_obs

        return {
            "infrastructure_state": infr_obs,
            "jobs_waiting_state": jobs_obs,
        }

    def _parse_step_info(self, raw_info: dict) -> dict:
        """Convert raw gRPC step info to job placement info dict."""
        return {
            "jobs_waiting": raw_info.get("jobs_waiting"),
            "jobs_placed": raw_info.get("jobs_placed"),
            "jobs_placed_ratio": raw_info.get("jobs_placed_ratio"),
            "quality_ratio": raw_info.get("quality_ratio"),
            "deadline_violation_ratio": raw_info.get("deadline_violation_ratio"),
            "job_wait_time": raw_info.get("job_wait_time"),
            "is_valid": raw_info.get("is_valid"),
        }

