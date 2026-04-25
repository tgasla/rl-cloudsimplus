import gymnasium as gym
import numpy as np

from gymnasium import spaces
from ..cloud_sim_grpc_client import CloudSimGrpcClient


# Based on https://gymnasium.farama.org/api/env/
class GrpcMultiDC(gym.Env):
    """
    Gymnasium environment that bridges Stable Baselines3 to CloudSim Plus (multi-DC) via gRPC.

    Each instance connects to its own gRPC server running in a dedicated Java JVM
    subprocess. This enables true parallel simulation when used with SubprocVecEnv.

    Key differences from GrpcSingleDC:
    1. infrastructure_observation is flat [dc_id, dc_type, free_vmpes] per host (not a tree array)
    2. jobs_waiting_observation is a flat int array [cores, location, sensitivity, deadline] per job
    3. action space: list of DC IDs (one per waiting job), each action[i] = DC index to place job i
    4. Supports both RL mode and heuristic modes (earliest-shortest-to-most-free-dc, etc.)

    Args:
        params:        Simulation configuration dict
        jobs_as_json:  JSON string of CloudletDescriptor list
        host:          Hostname of the gRPC server (default: localhost)
        port:          TCP port of the gRPC server (default: 50051)
        render_mode:   Gymnasium render mode (default: None)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    DC_TYPE_IDS = {"cloud": 0, "edge": 1, "micro": 2}

    def __init__(
        self,
        params: dict,
        jobs_as_json: str = "[]",
        host: str = "localhost",
        port: int = 50051,
        render_mode: str = None,
    ):
        super(GrpcMultiDC, self).__init__()
        self.params = params
        self.render_mode = render_mode

        # gRPC client - connects to its own Java JVM
        self._client = CloudSimGrpcClient(host=host, port=port)
        self._sim_id = None

        # ── Extract params ──────────────────────────────────────────────────
        self.max_datacenters = params["max_datacenters"]
        self.max_hosts = params["max_hosts"]
        self.max_jobs_waiting = params["max_jobs_waiting"]
        self.max_pes_per_vm = params.get("max_pes_per_vm", params.get("max_host_pes", 16))
        self.cloudlet_to_dc_assignment_policy = params.get(
            "cloudlet_to_dc_assignment_policy", "rl"
        )

        # ── Observation spaces ───────────────────────────────────────────────
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
        # max_val is a reasonable upper bound for job attributes
        max_val = max(self.max_pes_per_vm, 1000)  # cores, location, sensitivity, deadline
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
        # For heuristic modes, the RL action is not used but the env still needs an action space
        self.action_space = spaces.MultiDiscrete(
            np.array([self.max_datacenters] * self.max_jobs_waiting)
        )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn("Invalid render mode. Allowed: ['human', 'ansi']")

        # ── Create simulation ───────────────────────────────────────────────
        import json
        from utils.misc import _params_to_java_format
        java_params = _params_to_java_format(params)
        self._sim_id = self._client.create_simulation(json.dumps(java_params), jobs_as_json)

    # ── Action masking ────────────────────────────────────────────────────────
    def action_masks(self) -> list[bool]:
        """
        Returns a mask for the action space.

        For RL mode, masks out invalid DC choices based on current infrastructure state.
        A DC choice is invalid if that DC has no free VMs with enough cores.

        Returns:
            List of booleans indicating which actions are valid.
        """
        # The action space is MultiDiscrete([max_datacenters] * max_jobs_waiting)
        # We need to return a flat list of masks for SB3's MultiBinary masking
        masks = []

        # Get current infrastructure observation to determine valid DC choices
        # Each job can be placed in any DC, but we mask based on DC being "available"
        # For now, we allow all DC choices (they will be validated in step())
        # This can be enhanced with per-job core requirements later
        for _ in range(self.max_jobs_waiting):
            masks.append([True] * self.max_datacenters)

        return [item for sublist in masks for item in sublist]

    # ── Observation helpers ───────────────────────────────────────────────────
    def _pad_observation(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        if len(obs) >= target_dim:
            return np.array(obs[:target_dim], dtype=np.int16)
        padded = np.zeros(target_dim, dtype=np.int16)
        padded[: len(obs)] = obs
        return padded

    def _get_observation(self, raw_obs: dict) -> dict:
        """
        Convert raw observation from gRPC to gymnasium observation dict.

        Args:
            raw_obs: dict with keys "infrastructure_observation" and "jobs_waiting_observation"

        Returns:
            dict with "infrastructure_state" and "jobs_waiting_state"
        """
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

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """
        Reset the simulation.

        Args:
            seed: Random seed (ignored, handled by Java side)
            options: Additional options

        Returns:
            obs: Observation dict
            info: Info dict with simulation state
        """
        super(GrpcMultiDC, self).reset()
        self.current_step = 0

        if seed is None:
            seed = 0

        raw_result = self._client.reset(self._sim_id, seed)
        raw_obs = raw_result.get("observation", {})
        raw_info = raw_result.get("info", {})
        obs = self._get_observation(raw_obs)
        info = self._parse_step_info(raw_info)
        return obs, info

    def step(self, action):
        """
        Execute one step in the simulation.

        Args:
            action: For RL mode, a list of DC indices (one per waiting job).
                    For heuristic modes, this argument is ignored.

        Returns:
            obs: Observation dict
            reward: Float reward
            terminated: Boolean indicating if episode is done
            truncated: Boolean indicating if episode was truncated
            info: Info dict with step statistics
        """
        self.current_step += 1

        # gRPC expects a plain list of ints
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)

        result = self._client.step(self._sim_id, action_list)
        raw_obs = result.get("observation", {})
        reward = result.get("reward", 0.0)
        terminated = result.get("terminated", False)
        truncated = result.get("truncated", False)
        raw_info = result.get("info", {})

        obs = self._get_observation(raw_obs)
        info = self._parse_step_info(raw_info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _parse_step_info(self, raw_info: dict) -> dict:
        """
        Parse step info from the raw info dict.

        The StepInfo from Java contains:
        - jobs_waiting: int
        - jobs_placed: int
        - jobs_placed_ratio: double
        - quality_ratio: double
        - deadline_violation_ratio: double
        - job_wait_time: list of double

        Args:
            raw_info: Raw info dict from gRPC

        Returns:
            Parsed info dict
        """
        return {
            "jobsWaiting": raw_info.get("jobs_waiting", 0),
            "jobsPlaced": raw_info.get("jobs_placed", 0),
            "jobsPlacedRatio": raw_info.get("jobs_placed_ratio", 0.0),
            "qualityRatio": raw_info.get("quality_ratio", 0.0),
            "deadlineViolationRatio": raw_info.get("deadline_violation_ratio", 0.0),
            "jobWaitTime": raw_info.get("job_wait_time", []),
            "isValid": raw_info.get("is_valid", True),
            # Snake case keys for callback compatibility
            "jobs_waiting": raw_info.get("jobs_waiting", 0),
            "jobs_placed": raw_info.get("jobs_placed", 0),
            "jobs_placed_ratio": raw_info.get("jobs_placed_ratio", 0.0),
            "quality_ratio": raw_info.get("quality_ratio", 0.0),
            "deadline_violation_ratio": raw_info.get("deadline_violation_ratio", 0.0),
            "job_wait_time": raw_info.get("job_wait_time", []),
        }

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render_mode not set")
            return
        if self.render_mode == "human":
            print("Render (gRPC): not yet implemented")
        elif self.render_mode == "ansi":
            return "Render not yet implemented"

    def close(self):
        if self._sim_id is not None:
            self._client.close(self._sim_id)
        self._client.close_channel()
        # Stop the Java JVM subprocess spawned by this worker
        if hasattr(self, "_java_proc") and self._java_proc is not None:
            proc = self._java_proc
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)