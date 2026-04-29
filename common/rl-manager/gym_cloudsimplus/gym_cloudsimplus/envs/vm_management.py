"""VmManagementEnv — VM lifecycle management RL environment.

Inherits from CloudSimBaseEnv which provides all shared gRPC wiring.
Concrete domain-specific implementation for VM management problem:
- Tree-array infrastructure observation
- [action_type, host_id, vm_id, vm_type] action space
- VM lifecycle control (create S/M/L VMs, destroy VMs, no-op)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .base import CloudSimBaseEnv


class VmManagementEnv(CloudSimBaseEnv):
    """
    VM management Gymnasium environment bridging Stable Baselines3 to
    CloudSim Plus via gRPC.

    The agent controls VM lifecycle: create small/medium/large VMs on hosts,
    destroy existing VMs. Observation is a tree-array of the infrastructure
    (DC → hosts → VMs → jobs).

    Action space: MultiDiscrete([3, max_hosts, max_vms, 3])
        [action_type, host_id, vm_id, vm_type]
        action_type: 0=noop, 1=create, 2=destroy
        vm_type: 0=small, 1=medium, 2=large

    Inherits from CloudSimBaseEnv:
        - gRPC client (_client)
        - _sim_id, _rl_problem
        - reset(), step(), close(), ping()
        - _pad_observation()
    """

    VM_CORES = [2, 4, 8]  # small, medium, large

    def __init__(
        self,
        params: dict,
        jobs_as_json: str = "[]",
        host: str = "localhost",
        port: int = 50051,
        render_mode: str = "ansi",
    ):
        # Initialize base class (sets up _client, _sim_id=None, _rl_problem=None)
        super().__init__(params, jobs_as_json, host, port, render_mode)

        # Domain-specific RL problem type
        self._rl_problem = "vm_management"

        # ── Domain-specific fields ─────────────────────────────────────────────
        self.vm_allocation_policy = params["vm_allocation_policy"]
        self.host_count = params["host_count"]
        self.host_pes = params["host_pes"]
        self.small_vm_pes = params["small_vm_pes"]
        self.large_vm_multiplier = params["large_vm_multiplier"]

        self.reward_job_wait_coef = params["reward_job_wait_coef"]
        self.reward_running_vm_cores_coef = params["reward_running_vm_cores_coef"]
        self.reward_unutilized_vm_cores_coef = params["reward_unutilized_vm_cores_coef"]

        self.max_hosts = params["max_hosts"]
        self.action_types_count = 3  # noop, create vm, destroy vm
        self.vm_types_count = 3  # small, medium, large
        self.min_job_pes = 1
        self.large_vm_pes = self.small_vm_pes * self.large_vm_multiplier
        self.max_vms = self.max_hosts * int(self.host_pes) // int(self.small_vm_pes)
        self.max_jobs = self.max_hosts * int(self.host_pes) // self.min_job_pes

        # VM tracking for action masks
        self.host_cores_utilized = None  # initialized in reset()
        self.vms_running = 0

        # ── Action space ──────────────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete(
            np.array(
                [
                    self.action_types_count,
                    self.max_hosts,
                    self.max_vms,
                    self.vm_types_count,
                ]
            )
        )

        # ── Observation spaces ─────────────────────────────────────────────────
        # Tree-array: 2 * (1 + max_hosts + max_vms + max_jobs) values
        max_tree_nodes = 1 + self.max_hosts + self.max_vms + self.max_jobs
        self.infr_obs_length = 2 * max_tree_nodes

        max_pes_per_node = self.max_hosts * self.host_pes
        self.infr_obs_space = spaces.Box(
            low=0,
            high=max_pes_per_node,
            shape=(self.infr_obs_length,),
            dtype=np.int16,
        )
        self.job_cores_waiting_obs_space = spaces.Box(
            low=0,
            high=self.large_vm_pes + 1,
            shape=(1,),
            dtype=np.int16,
        )
        self.observation_space = spaces.Dict(
            {
                "infr_state": self.infr_obs_space,
                "job_cores_waiting_state": self.job_cores_waiting_obs_space,
            }
        )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                "Invalid render mode. Allowed: ['human', 'ansi']"
            )
        self.render_mode = render_mode

        # ── Create simulation (CloudSimBaseEnv has _client and _sim_id ready) ─
        import json
        self._sim_id = self._client.create_simulation(
            json.dumps(params), jobs_as_json
        )

    # ── CloudSimBaseEnv abstract methods ───────────────────────────────────────

    def _detect_rl_problem(self) -> str:
        return "vm_management"

    def action_masks(self) -> list[bool]:
        """Action mask for VM management action space."""
        if self.host_cores_utilized is None:
            return [True] * (3 + self.max_hosts + self.max_vms + 3)

        host_cores_utilized_sum = np.sum(self.host_cores_utilized)
        current_max_vms = self.host_count * int(self.host_pes) // int(self.small_vm_pes)

        if host_cores_utilized_sum == 0:  # no VMs running
            action_type_mask = [True, True, False]
            host_mask = [True] * self.host_count + [False] * (self.max_hosts - self.host_count)
            vm_mask = [True] + [False] * (self.max_vms - 1)
            vm_type_mask = [True, True, False]
        elif self.vms_running == current_max_vms:  # all VMs running
            action_type_mask = [True, False, True]
            host_mask = [True] + [False] * (self.max_hosts - 1)
            vm_mask = [True] * current_max_vms + [False] * (self.max_vms - current_max_vms)
            vm_type_mask = [True, False, False]
        elif all(self.host_pes - num < self.VM_CORES[0] for num in self.host_cores_utilized):
            action_type_mask = [True, False, True]
            host_mask = [True] + [False] * (self.max_hosts - 1)
            vm_mask = [True] * self.vms_running + [False] * (self.max_vms - self.vms_running)
            vm_type_mask = [True, False, False]
        elif all(self.host_pes - num < self.VM_CORES[1] for num in self.host_cores_utilized):
            action_type_mask = [True, True, True]
            host_mask = [True] * self.host_count + [False] * (self.max_hosts - self.host_count)
            vm_mask = [True] * self.vms_running + [False] * (self.max_vms - self.vms_running)
            vm_type_mask = [True, False, False]
        elif all(self.host_pes - num < self.VM_CORES[2] for num in self.host_cores_utilized):
            action_type_mask = [True, True, True]
            host_mask = [True] * self.host_count + [False] * (self.max_hosts - self.host_count)
            vm_mask = [True] * self.vms_running + [False] * (self.max_vms - self.vms_running)
            vm_type_mask = [True, True, False]
        else:  # common case
            full_host_indices = [
                i for i, num in enumerate(self.host_cores_utilized)
                if self.host_pes - num < self.VM_CORES[0]
            ]
            action_type_mask = [True, True, True]
            host_mask = [False if i in full_host_indices else True for i in range(self.host_count)]
            host_mask += [False] * (self.max_hosts - self.host_count)
            vm_mask = [True] * self.vms_running + [False] * (self.max_vms - self.vms_running)
            vm_type_mask = [True, True, True]

        masks = [action_type_mask, host_mask, vm_mask, vm_type_mask]
        return [item for sublist in masks for item in sublist]

    def _get_observation(self, raw_obs: dict) -> dict:
        """Convert raw gRPC observation to VM management gymnasium obs dict."""
        infr_obs = np.array(raw_obs["infr_state"], dtype=np.int16)
        infr_obs = self._pad_observation(infr_obs, self.infr_obs_length)
        # Pad job_cores_waiting_state to (1,) so VecEnv stacking produces (n_envs, 1)
        raw_jcw = raw_obs["job_cores_waiting_state"]
        jcw = np.array(raw_jcw, dtype=np.int16)
        if jcw.shape == ():  # scalar 0D
            jcw = jcw.reshape(1)
        return {
            "infr_state": infr_obs,
            "job_cores_waiting_state": jcw,
        }

    def _parse_step_info(self, raw_info: dict) -> dict:
        """Convert raw gRPC step info to VM management info dict."""
        return {
            "job_wait_reward": raw_info.get("job_wait_reward", 0.0),
            "running_vm_cores_reward": raw_info.get("running_vm_cores_reward", 0.0),
            "unutilized_vm_cores_reward": raw_info.get("unutilized_vm_cores_reward", 0.0),
            "invalid_reward": raw_info.get("invalid_reward", 0.0),
            "is_valid": raw_info.get("is_valid", True),
            "host_affected": raw_info.get("host_affected", 0),
            "cores_changed": raw_info.get("cores_changed", 0),
        }

    # ── reset() override — needs VM tracking init ──────────────────────────────
    def reset(self, seed=None, options=None):
        """Reset the simulation. Also re-initializes VM tracking state."""
        obs, info = super().reset(seed=seed, options=options)
        # Initialize VM tracking after base reset (base calls gRPC reset)
        self.host_cores_utilized = np.zeros(self.max_hosts, dtype=np.int32)
        self.vms_running = 0
        return obs, info

    # ── step() override — needs VM tracking update ────────────────────────────
    def step(self, action):
        """Execute one step. Also updates VM tracking for action masks."""
        obs, reward, terminated, truncated, info = super().step(action)
        # Update VM tracking for next action mask
        if info.get("is_valid"):
            action_list = action.tolist() if hasattr(action, "tolist") else list(action)
            if action_list[0] == 1:  # create VM
                host_id = action_list[1]
                vm_type = action_list[3]
                self.host_cores_utilized[host_id] += self.VM_CORES[vm_type]
                self.vms_running += 1
            elif action_list[0] == 2:  # destroy VM
                host_id = info["host_affected"]
                self.host_cores_utilized[host_id] -= info["cores_changed"]
                self.vms_running -= 1
        return obs, reward, terminated, truncated, info

    def _pad_observation(self, obs, target_dim: int) -> np.ndarray:
        """Pad observation array to target dimension."""
        if len(obs) >= target_dim:
            return np.array(obs[:target_dim], dtype=np.int16)
        padded = np.zeros(target_dim, dtype=np.int16)
        padded[: len(obs)] = obs
        return padded