import gymnasium as gym
import numpy as np

from gymnasium import spaces
from ..cloud_sim_grpc_client import CloudSimGrpcClient


# Based on https://gymnasium.farama.org/api/env/
class GrpcSingleDC(gym.Env):
    """
    Gymnasium environment that bridges Stable Baselines3 to CloudSim Plus via gRPC.

    Unlike the Py4J-based SingleDC, each instance connects to its own gRPC server
    running in a dedicated Java JVM subprocess. This enables true parallel simulation
    when used with SubprocVecEnv.

    Args:
        params:        Simulation configuration dict (same as SingleDC)
        jobs_as_json: JSON string of CloudletDescriptor list
        host:          Hostname of the gRPC server (default: localhost)
        port:          TCP port of the gRPC server (default: 50051)
        render_mode:   Gymnasium render mode (default: "ansi")
    """

    metadata = {"render_modes": ["human", "ansi"]}
    VM_CORES = [2, 4, 8]

    def __init__(
        self,
        params: dict,
        jobs_as_json: str = "[]",
        host: str = "localhost",
        port: int = 50051,
        render_mode: str = "ansi",
    ):
        super(GrpcSingleDC, self).__init__()
        self.params = params

        # gRPC client - connects to its own Java JVM
        self._client = CloudSimGrpcClient(host=host, port=port)
        self._sim_id = None

        # ── Mirror all fields from SingleDC ──────────────────────────────────
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

        # self.infr_obs_length = max_tree_nodes = (1 + self.max_hosts + self.max_vms + self.max_jobs)
        # with the term "node" i refer to a tree node which can be a DC, a host, a VM or a job
        # the node that has the most pes is the DC because it aggregates all the hosts's pes
        # we do (max_pes_per_node + 1) because the DC can have 0 pes as well.
        # So, these are the different possible values a tree node can take in the infrastructure length,
        # 0, 1, 2, ..., max_pes_per_node -> max_pes_per_node + 1 distinct values
        # we multiply by np.ones(self.infr_obs_length) a vector of ones with the same length as the
        # infrastructure length.
        # So, the vector will look like self.infr_obs_space = [max_pes_per_node + 1, max_pes_per_node + 1, ..., max_pes_per_node + 1]
        # where each element is the number of pes of a node in the infrastructure or the number of children of a node in the infrastructure.
        # Here, I forgot to add the number of children of a node in the infrastructure.
        # So in reality the self.infr_obs_length should be 2 * (1 + self.max_hosts + self.max_vms + self.max_jobs)
        # because the infrastruture state is eventually a pre-order traversal of the tree of nodes, writing the pes of a node first,
        # and then the number of children of that node.
        # Now, the maximum value for all elements in the vector is max_pes_per_node + 1. This, assumes that always:
        # 1. The number of children of a node is less than or equal to max_pes_per_node.
        # 2. The number of pes of a node is less than or equal to max_pes_per_node.
        # Corrected below!:
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

        # ── Create simulation ───────────────────────────────────────────────
        import json
        self._sim_id = self._client.create_simulation(json.dumps(params), jobs_as_json)

    # ── Action masking ────────────────────────────────────────────────────────
    def action_masks(self) -> list[bool]:
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

    # ── Observation helpers ───────────────────────────────────────────────────
    def _pad_observation(self, obs, target_dim: int) -> np.ndarray:
        if len(obs) >= target_dim:
            return np.array(obs[:target_dim], dtype=np.int16)
        padded = np.zeros(target_dim, dtype=np.int16)
        padded[: len(obs)] = obs
        return padded

    def _get_observation(self, raw_obs: dict) -> dict:
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

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super(GrpcSingleDC, self).reset()
        self.current_step = 0
        self.host_cores_utilized = np.zeros(self.max_hosts, dtype=np.int32)
        self.vms_running = 0

        if seed is None:
            seed = 0

        raw_result = self._client.reset(self._sim_id, seed)
        obs = self._get_observation(raw_result.get("observation", {}))
        info = raw_result.get("info", {})
        return obs, info

    def step(self, action):
        self.current_step += 1

        # gRPC expects a plain list of ints
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)

        result = self._client.step(self._sim_id, action_list)

        obs = self._get_observation(result.get("observation", {}))
        reward = result.get("reward", 0.0)
        terminated = result.get("terminated", False)
        truncated = result.get("truncated", False)
        info = result.get("info", {})

        if self.render_mode == "human":
            self.render()

        # Update VM tracking for action masks
        if info.get("is_valid"):
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

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render_mode not set")
            return
        # gRPC render not yet implemented on the Java side
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
