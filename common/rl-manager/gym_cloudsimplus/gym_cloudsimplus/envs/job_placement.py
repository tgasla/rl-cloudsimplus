"""JobPlacementEnv — job-to-datacenter placement RL environment.

Inherits from CloudSimBaseEnv which provides all shared gRPC wiring.
Concrete domain-specific implementation for job placement problem:
- Flat per-host [dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts] observation
- Per-job [cores, nominal_runtime_ref, time_to_due, s0, s1, s2] observation
- Per-(job, action) reachability mask observation
- [dc_index, dc_index, ...] action space (one per waiting job)
"""

import numpy as np
from gymnasium import spaces

from .base import CloudSimBaseEnv


# Java gives every cloudlet a file size of one MTU and places it only on a VM whose storage
# holds it (Vm.isSuitableForCloudlet).
CLOUDLET_FILE_SIZE = 1500


def _check_one_vm_per_host(datacenters: list) -> None:
    """The observation reports one capacity per host and the action mask is built from it,
    while Java checks each VM; the two only agree when one VM fills each host."""
    for dc in datacenters:
        for host in dc["hosts"]:
            vms = host["vms"]
            if len(vms) != 1 or vms[0].get("amount", 1) != 1 or vms[0]["pes"] != host["pes"]:
                raise ValueError(f"{dc['name']}: every host must run exactly one VM as large as "
                                 f"the host, got {vms}")
            if vms[0]["size"] < CLOUDLET_FILE_SIZE:
                raise ValueError(f"{dc['name']}: VM size {vms[0]['size']} cannot hold a cloudlet "
                                 f"({CLOUDLET_FILE_SIZE})")


class JobPlacementEnv(CloudSimBaseEnv):
    """
    Job placement Gymnasium environment bridging Stable Baselines3 to
    CloudSim Plus via gRPC.

    The agent decides which datacenter to place each waiting job into.
    Observation (all flat):
        infrastructure_state: [dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts]
                              per host slot (total_hosts slots, zero-padded)
        jobs_waiting_state:   [cores, nominal_runtime_ref, time_to_due, s0, s1, s2] per job
                              slot (max_jobs_waiting slots, zero-padded); s* one-hot the
                              delay sensitivity (tolerant, moderate, critical)
        reach_mask:           [max_jobs_waiting, max_datacenters] — 1 where placing job slot
                              j via action k is legal under the topology; column 0 is the
                              no-op, the only legal action for a padding slot

    Action space: MultiDiscrete([max_datacenters] * max_jobs_waiting)
        action[i] = DC index to place job i (0 = no-op, k = datacenters[k - 1])

    Inherits from CloudSimBaseEnv:
        - gRPC client (_client)
        - _sim_id, _rl_problem
        - reset(), step(), close(), ping()
        - _pad_observation()
    """

    DC_TYPE_IDS = {"cloud": 1, "edge": 2, "micro": 3}  # 0 = padding slot; must match Java getDcTypeIdFromStr
    HOST_OBS_FEATURES = 5     # dc_id, dc_type, vm_capacity_pes, free_pes, backlog_core_ts — must match Java WrappedSimulation.HOST_OBS_FEATURES
    JOB_OBS_FEATURES = 6      # policy-visible features per job: cores, nominal_runtime_ref, time_to_due, s0, s1, s2
    _JOB_WIRE_FEATURES = 7    # gRPC wire format: the policy features plus location at index 1 — must match Java CloudSimProxy.JOB_OBS_FEATURES
    _LOCATION_COL = 1         # location is stripped from the policy obs; it drives reach_mask
    _VM_CAPACITY_COL = 2
    _STEP_INFO_KEYS = (
        "jobs_waiting", "jobs_placed", "jobs_placed_ratio", "job_wait_time",
        "sla_value_realized", "sla_penalty_paid", "resource_cost", "jobs_met", "jobs_violated",
        "jobs_expired_unplaced", "potential", "offered_value", "unshaped_reward",
    )

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
        self.max_host_pes = params["max_host_pes"]
        self.max_job_pes = params["max_job_pes"]
        self.cloudlet_to_dc_mapping = params.get("cloudlet_to_dc_mapping", "rl")

        # ── Observation spaces ─────────────────────────────────────────────────
        # total_hosts is pinned in config, not derived from max_hosts * max_datacenters, so
        # the observation shape does not move with the per-DC host cap.
        self.total_hosts = params["total_hosts"]
        datacenters = params["datacenters"]
        n_hosts = sum(h.get("amount", 1) for dc in datacenters for h in dc.get("hosts", []))
        if n_hosts > self.total_hosts:
            raise ValueError(
                f"topology has {n_hosts} hosts but total_hosts={self.total_hosts}"
            )
        unbounded = np.iinfo(np.int32).max

        self.infr_obs_length = self.HOST_OBS_FEATURES * self.total_hosts
        host_high = np.array([
            self.max_datacenters - 1,   # dc_id
            len(self.DC_TYPE_IDS),      # dc_type
            self.max_host_pes,          # vm_capacity_pes
            self.max_host_pes,          # free_pes
            unbounded,                  # backlog_core_ts
        ], dtype=np.int32)
        self.infr_obs_space = spaces.Box(
            low=0,
            high=np.tile(host_high, self.total_hosts),
            shape=(self.infr_obs_length,),
            dtype=np.int32,
        )

        self.job_obs_length = self.JOB_OBS_FEATURES * self.max_jobs_waiting
        job_high = np.array(
            [self.max_job_pes, unbounded, unbounded, 1, 1, 1], dtype=np.int32
        )
        self.job_waiting_obs_space = spaces.Box(
            low=0,
            high=np.tile(job_high, self.max_jobs_waiting),
            shape=(self.job_obs_length,),
            dtype=np.int32,
        )

        self.reach_obs_length = self.max_jobs_waiting * self.max_datacenters
        self.reach_obs_space = spaces.Box(
            low=0, high=1, shape=(self.reach_obs_length,), dtype=np.int8
        )

        self.observation_space = spaces.Dict(
            {
                "infrastructure_state": self.infr_obs_space,
                "jobs_waiting_state": self.job_waiting_obs_space,
                "reach_mask": self.reach_obs_space,
            }
        )

        # ── Action space ──────────────────────────────────────────────────────
        # For RL mode: MultiDiscrete([max_datacenters] * max_jobs_waiting)
        # Only the first N elements (N = number of waiting jobs) are valid
        self.action_space = spaces.MultiDiscrete(
            np.array([self.max_datacenters] * self.max_jobs_waiting)
        )

        # ── Last observation cache (used by action_masks) ─────────────────────
        self._last_infr_obs = np.zeros(self.infr_obs_length, dtype=np.int32)
        self._last_jobs_obs = np.zeros(self.job_obs_length, dtype=np.int32)
        self._last_reach = np.zeros((self.max_jobs_waiting, self.max_datacenters), dtype=bool)

        # ── Per-episode instances (set_level_stream); None replays the creation jobs ──
        self._levels = None
        self._level_sampler = None
        self._level_id = None

        # ── Permutation stress-test flag ───────────────────────────────────────
        # When True, DC host groups are shuffled randomly at each observation.
        # Permutation-invariant extractors (type_stratified, hybrid, spane,
        # attention_pooling) should maintain performance; positionally-biased
        # extractors (euromlsys flat MLP) will degrade.
        self.permute_dcs = params.get("permute_dcs", False)
        # Its own stream per worker: workers step in threads, and a shared global stream would
        # be consumed in scheduling order.
        self._rng = np.random.default_rng([params.get("seed", 0), params.get("worker_rank", 0)])

        # ── Topology connectivity mask ────────────────────────────────────────
        # Precomputed static table: _location_valid_dc_mask[loc, action] = True
        # means a job originating from DC 'loc' (0-based) may be placed via
        # action 'action' (1-based, 0 = no-op).
        # Action k corresponds to datacenters[k-1]. Built once from connect_to
        # after name→index translation in entrypoint.py.
        # Action 0 is the no-op, so only max_datacenters - 1 real DCs are addressable.
        # A bigger topology would have its last DCs silently unmaskable (dead).
        _check_one_vm_per_host(datacenters)
        n_dcs = len(datacenters)
        if n_dcs > self.max_datacenters - 1:
            raise ValueError(
                f"topology has {n_dcs} datacenters but max_datacenters={self.max_datacenters} "
                f"addresses only {self.max_datacenters - 1} (action 0 is the no-op)"
            )
        self._location_valid_dc_mask = self._build_location_mask(datacenters)

        # ── Create simulation (CloudSimBaseEnv has _client and _sim_id ready) ─
        import json
        self._sim_id = self._client.create_simulation(
            json.dumps(params), jobs_as_json
        )

    def set_level_stream(self, levels, sampler) -> None:
        """Play a new problem instance at every reset: `sampler.next()` picks a level id and
        `levels.jobs_json(level_id)` its jobs. SB3's auto-reset passes no options, so the
        environment has to own the stream rather than have the caller ship jobs."""
        self._levels = levels
        self._level_sampler = sampler

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        self._level_id = None
        if "jobs_json" not in options and self._level_sampler is not None:
            self._level_id = self._level_sampler.next()
            options["jobs_json"] = self._levels.jobs_json(self._level_id)
        return super().reset(seed=seed, options=options)

    # ── CloudSimBaseEnv abstract methods ───────────────────────────────────────

    def _build_location_mask(self, datacenters: list) -> np.ndarray:
        """Build static [n_dc, max_datacenters] bool mask from topology connect_to.

        mask[loc, action] = True means a job originating at DC loc (0-based)
        can be placed via action (action=0 no-op excluded — handled separately).
        DCs with no connect_to (cloud/edge destinations) allow every real DC.
        """
        n_dc = len(datacenters)
        mask = np.zeros((n_dc, self.max_datacenters), dtype=bool)
        for loc_idx, dc in enumerate(datacenters):
            connect_to = dc.get("connect_to", [])
            if not connect_to:
                # Destination DC (cloud/edge): no origin restriction
                mask[loc_idx, 1:n_dc + 1] = True
            else:
                # Micro DC: can place at itself or at explicitly connected DCs
                for dest_idx in [loc_idx] + list(connect_to):
                    action = dest_idx + 1  # obs_dc_id is 1-based
                    if action < self.max_datacenters:
                        mask[loc_idx, action] = True
        return mask

    def _permute_infr_obs(self, infr_obs: np.ndarray) -> np.ndarray:
        """Randomly shuffle DC host-groups in the flat infrastructure observation.

        Host rows start with dc_id. We group rows
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
        self._rng.shuffle(active_dcs)
        groups = [hosts[dc_ids == dc_id] for dc_id in active_dcs]
        padding = hosts[dc_ids == 0]
        parts = groups + ([padding] if len(padding) > 0 else [])
        return np.concatenate(parts, axis=0).flatten().astype(infr_obs.dtype)

    def action_masks(self) -> list[bool]:
        """Return action mask for MaskablePPO.

        A (job, action) pair is valid when the topology allows it (reach_mask) and the
        DC has a VM large enough to ever hold the job. A full DC stays valid: Java queues
        the job there, and the later finish is priced by the reward. This must match
        Java's VM selector. The no-op (action 0) is always valid, so every sub-space keeps
        at least one valid action; a padding job slot (cores == 0) can only take the no-op.

        obs_dc_id = cloudSim_dc_id - 1, which equals the agent action for that DC.
        """
        # Scatter-max: largest VM per DC action index in one vectorized pass.
        hosts = self._last_infr_obs.reshape(self.total_hosts, self.HOST_OBS_FEATURES)
        dc_ids = hosts[:, 0].astype(np.int64)
        capacity = hosts[:, self._VM_CAPACITY_COL].astype(np.int64)
        real = dc_ids > 0
        dc_max_capacity = np.zeros(self.max_datacenters, dtype=np.int64)
        np.maximum.at(dc_max_capacity, dc_ids[real], capacity[real])

        # Broadcast [max_jobs, 1] cores against [1, max_datacenters] capacity.
        cores = self._last_jobs_obs.reshape(self.max_jobs_waiting, self.JOB_OBS_FEATURES)[:, 0]
        mask_matrix = self._last_reach & (dc_max_capacity[np.newaxis, :] >= cores[:, np.newaxis])
        mask_matrix[:, 0] = True  # no-op
        return mask_matrix.ravel().tolist()

    def _reach_matrix(self, cores: np.ndarray, locations: np.ndarray) -> np.ndarray:
        """[max_jobs_waiting, max_datacenters] topology legality per job slot and action."""
        reach = np.zeros((self.max_jobs_waiting, self.max_datacenters), dtype=bool)
        real = cores > 0
        if (locations[real] >= len(self._location_valid_dc_mask)).any():
            raise ValueError(
                f"job location out of range for {len(self._location_valid_dc_mask)} "
                f"datacenters: {locations[real].max()}"
            )
        reach[real] = self._location_valid_dc_mask[locations[real]]
        reach[:, 0] = True  # no-op
        return reach

    def _get_observation(self, raw_obs: dict) -> dict:
        """Convert raw gRPC observation to job placement gymnasium obs dict."""
        infr_obs = np.array(raw_obs.get("infrastructure_observation"), dtype=np.int32)
        infr_obs = self._pad_observation(infr_obs, self.infr_obs_length)
        if self.permute_dcs:
            infr_obs = self._permute_infr_obs(infr_obs)
        self._last_infr_obs = infr_obs

        # Jobs: strip location before exposing to the policy; it only drives reach_mask.
        raw_jobs = np.array(raw_obs.get("secondary_observation"), dtype=np.int32)
        raw_jobs = self._pad_observation(raw_jobs, self._JOB_WIRE_FEATURES * self.max_jobs_waiting)
        raw_feats = raw_jobs.reshape(self.max_jobs_waiting, self._JOB_WIRE_FEATURES)
        locations = raw_feats[:, self._LOCATION_COL].astype(np.int64)
        jobs_obs = np.delete(raw_feats, self._LOCATION_COL, axis=1).ravel()
        self._last_jobs_obs = jobs_obs
        self._last_reach = self._reach_matrix(jobs_obs[:: self.JOB_OBS_FEATURES], locations)

        return {
            "infrastructure_state": infr_obs,
            "jobs_waiting_state": jobs_obs,
            "reach_mask": self._last_reach.ravel().astype(np.int8),
        }

    def _parse_step_info(self, raw_info: dict) -> dict:
        """Convert raw gRPC step info to job placement info dict."""
        info = {key: raw_info.get(key) for key in self._STEP_INFO_KEYS}
        info["level_id"] = self._level_id
        return info
