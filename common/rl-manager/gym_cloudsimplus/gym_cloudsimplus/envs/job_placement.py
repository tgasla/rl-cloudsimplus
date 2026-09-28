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
    HOST_OBS_FEATURES = 3     # dc_id, dc_type, free_vmpes — must match Java WrappedSimulation.HOST_OBS_FEATURES
    JOB_OBS_FEATURES = 3      # policy-visible features per job: cores, delaySensitivity, deadline
    _JOB_GGRPC_FEATURES = 4   # gRPC wire format: [cores, location, delaySensitivity, deadline] — location (idx 1) cached for masking only

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
        self.infr_obs_length = self.HOST_OBS_FEATURES * total_hosts
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
        # Job locations cached separately — read from gRPC wire obs (index 1) but
        # not exposed to the policy (location has zero reward correlation; masking
        # enforces connectivity structurally).
        self._last_locations = np.zeros(self.max_jobs_waiting, dtype=np.int64)

        # ── Permutation stress-test flag ───────────────────────────────────────
        # When True, DC host groups are shuffled randomly at each observation.
        # Permutation-invariant extractors (type_stratified, hybrid, spane,
        # attention_pooling) should maintain performance; positionally-biased
        # extractors (euromlsys flat MLP) will degrade.
        self.permute_dcs = params.get("permute_dcs", False)

        # ── Topology connectivity mask ────────────────────────────────────────
        # Precomputed static table: _location_valid_dc_mask[loc, action] = True
        # means a job originating from DC 'loc' (0-based) may be placed via
        # action 'action' (1-based, 0 = no-op).
        # Action k corresponds to datacenters[k-1]. Built once from connect_to
        # after name→index translation in entrypoint.py.
        # Action 0 is the no-op, so only max_datacenters - 1 real DCs are addressable.
        # A bigger topology would have its last DCs silently unmaskable (dead).
        n_dcs = len(params.get("datacenters", []))
        if n_dcs > self.max_datacenters - 1:
            raise ValueError(
                f"topology has {n_dcs} datacenters but max_datacenters={self.max_datacenters} "
                f"addresses only {self.max_datacenters - 1} (action 0 is the no-op)"
            )
        self._location_valid_dc_mask = self._build_location_mask(
            params.get("datacenters", [])
        )

        # ── Create simulation (CloudSimBaseEnv has _client and _sim_id ready) ─
        import json
        self._sim_id = self._client.create_simulation(
            json.dumps(params), jobs_as_json
        )

    # ── CloudSimBaseEnv abstract methods ───────────────────────────────────────

    def _build_location_mask(self, datacenters: list) -> np.ndarray | None:
        """Build static [n_dc, max_datacenters] bool mask from topology connect_to.

        mask[loc, action] = True means a job originating at DC loc (0-based)
        can be placed via action (action=0 no-op excluded — handled separately).
        DCs with no connect_to (cloud/edge destinations) allow all actions.
        """
        if not datacenters:
            return None
        n_dc = len(datacenters)
        mask = np.zeros((n_dc, self.max_datacenters), dtype=bool)
        for loc_idx, dc in enumerate(datacenters):
            connect_to = dc.get("connect_to", [])
            if not connect_to:
                # Destination DC (cloud/edge): no origin restriction
                mask[loc_idx, :] = True
            else:
                # Micro DC: can place at itself or at explicitly connected DCs
                for dest_idx in [loc_idx] + list(connect_to):
                    action = dest_idx + 1  # obs_dc_id is 1-based
                    if action < self.max_datacenters:
                        mask[loc_idx, action] = True
        return mask

    def _permute_infr_obs(self, infr_obs: np.ndarray) -> np.ndarray:
        """Randomly shuffle DC host-groups in the flat infrastructure observation.

        Host features are [dc_id, dc_type, free_vmpes] per row. We group rows
        by dc_id value, shuffle the group order, then reassemble — so dc_id
        values are preserved but their positions in the flat array change.

        Permutation-invariant extractors (type_stratified, hybrid, spane,
        attention_pooling) are unaffected because they read dc_id by value via
        scatter_add. The euromlsys flat MLP reads by position and degrades.
        """
        hosts = infr_obs.reshape(-1, self.HOST_OBS_FEATURES)
        dc_ids = hosts[:, 0]
        active_dcs = list({int(d) for d in dc_ids if d > 0})
        if len(active_dcs) <= 1:
            return infr_obs
        np.random.shuffle(active_dcs)
        groups = [hosts[dc_ids == dc_id] for dc_id in active_dcs]
        padding = hosts[dc_ids == 0]
        parts = groups + ([padding] if len(padding) > 0 else [])
        return np.concatenate(parts, axis=0).flatten().astype(infr_obs.dtype)

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
        n_hosts = self.infr_obs_length // self.HOST_OBS_FEATURES
        hosts = infr_obs.reshape(n_hosts, self.HOST_OBS_FEATURES)
        dc_ids = hosts[:, 0].astype(np.int64)
        free_pes = hosts[:, 2].astype(np.int64)
        valid = (dc_ids > 0) & (dc_ids < self.max_datacenters)
        dc_max_free = np.zeros(self.max_datacenters, dtype=np.int64)
        np.maximum.at(dc_max_free, dc_ids[valid], free_pes[valid])

        # Broadcast [max_jobs, 1] cores against [1, max_datacenters] capacity.
        job_feats = jobs_obs.reshape(self.max_jobs_waiting, self.JOB_OBS_FEATURES)
        cores = job_feats[:, 0].astype(np.int64)
        mask_matrix = dc_max_free[np.newaxis, :] >= cores[:, np.newaxis]  # [J, DC]

        # Connectivity constraint: job can only be placed at its origin DC or
        # DCs reachable via connect_to. Padding jobs (cores=0) are unrestricted.
        if self._location_valid_dc_mask is not None:
            locations = self._last_locations.clip(
                0, len(self._location_valid_dc_mask) - 1
            )
            conn_mask = self._location_valid_dc_mask[locations]  # [J, max_dc]
            conn_mask[cores == 0] = True  # padding slots: unrestricted
            mask_matrix &= conn_mask

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
        if self.permute_dcs:
            infr_obs = self._permute_infr_obs(infr_obs)
        self._last_infr_obs = infr_obs

        # Jobs waiting: gRPC sends [cores, location, sensitivity, deadline] per job (4 features).
        # Strip location before exposing to the policy; cache it for action masking.
        raw_jobs = np.array(raw_obs.get("secondary_observation"), dtype=np.int16)
        raw_jobs = self._pad_observation(raw_jobs, self._JOB_GGRPC_FEATURES * self.max_jobs_waiting)
        raw_feats = raw_jobs.reshape(self.max_jobs_waiting, self._JOB_GGRPC_FEATURES)
        self._last_locations = raw_feats[:, 1].astype(np.int64)  # location — masking only
        # Policy obs: [cores, sensitivity, deadline] (columns 0, 2, 3)
        jobs_obs = np.concatenate([raw_feats[:, :1], raw_feats[:, 2:]], axis=1).flatten().astype(np.int16)
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

