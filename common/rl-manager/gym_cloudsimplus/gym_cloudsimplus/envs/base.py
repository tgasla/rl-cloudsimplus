"""CloudSimBaseEnv — shared base for CloudSim Gymnasium environments.

Contains ALL shared gRPC wiring: channel lifecycle, simulation lifecycle,
observation padding, RL problem detection, and step/reset/close/ping methods.

Subclasses implement domain-specific logic:
- VmManagementEnv: tree-array observation, VM-lifecycle action space
- JobPlacementEnv: flat per-host observation, job-to-DC placement action space
"""

import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from gymnasium import spaces

from ..cloud_sim_grpc_client import CloudSimGrpcClient, _detect_rl_problem


class CloudSimBaseEnv(gym.Env, ABC):
    """Shared base class for CloudSim gRPC environments.

    Handles:
    - gRPC channel and simulation lifecycle
    - RL problem type detection
    - Observation padding utility
    - batch_step support for SubprocVecEnv

    Subclasses MUST set:
    - self._rl_problem in __init__ (set by super().__init__ caller)
    - self.observation_space (gymnasium Space)
    - self.action_space (gymnasium Space)
    - self._get_observation(raw_obs) returning the domain-specific obs dict
    - self._parse_step_info(raw_info) returning domain-specific info dict
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        params: dict,
        jobs_as_json: str = "[]",
        host: str = "localhost",
        port: int = 50051,
        render_mode: str = None,
    ):
        super().__init__()
        self.params = params
        self.render_mode = render_mode
        self._current_step = 0

        # gRPC client — no paper= parameter, uses unified proto
        self._client = CloudSimGrpcClient(host=host, port=port)
        self._sim_id = None

        # RL problem type MUST be set by subclass before reset/step are called
        self._rl_problem = None  # "vm_management" or "job_placement"

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                f"Invalid render mode '{render_mode}'. Allowed: {self.metadata['render_modes']}"
            )

    @abstractmethod
    def _detect_rl_problem(self) -> str:
        """Detect RL problem from params. Subclasses can override."""
        return _detect_rl_problem(self.params)

    @abstractmethod
    def _get_observation(self, raw_obs: dict) -> dict:
        """Convert raw gRPC observation dict to gymnasium observation dict.

        Args:
            raw_obs: raw observation dict from gRPC response, keys vary by RL problem type

        Returns:
            domain-specific observation dict matching self.observation_space
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_step_info(self, raw_info: dict) -> dict:
        """Convert raw gRPC step info dict to gymnasium info dict.

        Args:
            raw_info: raw info dict from gRPC response, fields vary by RL problem type

        Returns:
            domain-specific info dict
        """
        raise NotImplementedError

    @abstractmethod
    def action_masks(self) -> list[bool]:
        """Return action masks for the current simulation state.

        Returns:
            flat list of booleans, one per discrete action element
        """
        raise NotImplementedError

    # ── Gymnasium API (override in subclass) ───────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset()
        self._current_step = 0
        if seed is None:
            seed = 0
        raw_result = self._client.reset(self._sim_id, seed, rl_problem=self._rl_problem)
        obs = self._get_observation(raw_result.get("observation", {}))
        raw_info = raw_result.get("info", {})
        info = self._parse_step_info(raw_info)
        return obs, info

    def step(self, action):
        self._current_step += 1
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        result = self._client.step(self._sim_id, action_list, rl_problem=self._rl_problem)
        obs = self._get_observation(result.get("observation", {}))
        reward = result.get("reward", 0.0)
        terminated = result.get("terminated", False)
        truncated = result.get("truncated", False)
        raw_info = result.get("info", {})
        info = self._parse_step_info(raw_info)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

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
        if hasattr(self, "_java_proc") and self._java_proc is not None:
            proc = self._java_proc
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)

    def ping(self) -> bool:
        """Health check against the gRPC server."""
        return self._client.ping()
