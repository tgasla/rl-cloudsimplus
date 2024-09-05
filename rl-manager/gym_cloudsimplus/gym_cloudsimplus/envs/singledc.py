from curses import raw
import gymnasium as gym
import os
import json
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters
import numpy as np

# TODO: the two environments should support both continuous and
# discrete action spaces


# Based on https://gymnasium.farama.org/api/env/
class SingleDC(gym.Env):
    """
    Action space - Continuous

    A vector of 2 continuous numbers

    The first number is in range [-1,1]
    Positive value - create a VM
    The range shows the id of host that the VM will be placed

    Negative value - destroy a VM
    The range shows the id of the VM that will be destroyed

    The second number is in range [0,1]
    It shows the size of the VM that will be created/destroyed

    Action space - Discrete

    A vector of 3 discrete numbers ...

    Observation space
    """

    metadata = {"render_modes": ["human", "ansi"]}
    parameters = GatewayParameters(address="gateway", auto_convert=True)

    def __init__(
        self,
        jobs_as_json="[]",
        render_mode="ansi",
    ):
        super(SingleDC, self).__init__()

        host_count = os.getenv("HOST_COUNT")
        host_pes = os.getenv("HOST_PES")
        small_vm_pes = os.getenv("SMALL_VM_PES")

        self.gateway = JavaGateway(gateway_parameters=self.parameters)
        self.simulation_environment = self.gateway.entry_point

        # TODO: have to define it in .env and pass it preperly in arg
        self.min_job_pes = 1
        self.max_vms_count = int(host_count) * int(host_pes) // int(small_vm_pes)
        self.max_jobs_count = self.max_vms_count * int(small_vm_pes) // self.min_job_pes
        self.observation_rows = (
            1 + int(host_count) + self.max_vms_count + self.max_jobs_count
        )
        self.observation_cols = 4

        # Old for continuous action space
        # self.action_space = spaces.Box(
        #     low=np.array([-1.0, 0.0]),
        #     high=np.array([1.0, 1.0]),
        #     shape=(2,),
        #     dtype=np.float32
        # )

        # New for discrete action space
        # [action, id, type^]
        # action = {0: do nothing, 1: create vm, 2: destroy vm}
        # type = {0: small, 1: medium, 2: large}
        # ^ needed only when action = 1
        self.action_space = spaces.MultiDiscrete(
            np.array([3, int(host_count), self.max_vms_count, 3])
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.observation_rows, self.observation_cols),
            dtype=np.float32,
        )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                "Invalid render mode" 'Render modes allowed: ["human" | "ansi"]'
            )

        self.render_mode = render_mode

        self.simulation_id = self.simulation_environment.createSimulation(jobs_as_json)

    def reset(self, seed=None, options=None):
        super(SingleDC, self).reset()

        result = self.simulation_environment.reset(self.simulation_id)
        self.simulation_environment.seed(self.simulation_id)

        raw_obs = result.getObs()
        obs = self._to_nparray(raw_obs)
        raw_info = result.getInfo()
        info = self._raw_info_to_dict(raw_info)

        return obs, info

    def step(self, action):
        # Py4J cannot translate np.array(dtype=np.float32) to java List<double>
        # Fix1: make it dtype=np.float64 and for some reason it works :)
        # Fix2: before sending it to java, convert it to python list first
        # Here, we adopt Fix2

        action = action.tolist()
        result = self.simulation_environment.step(self.simulation_id, action)

        reward = result.getReward()
        raw_info = result.getInfo()
        terminated = result.isTerminated()
        truncated = result.isTruncated()
        raw_obs = result.getObs()
        obs = self._to_nparray(raw_obs)

        info = self._raw_info_to_dict(raw_info)

        if self.render_mode == "human":
            self.render()

        return (obs, reward, terminated, truncated, info)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method "
                "without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="human")'
            )
            return
        # result is a string with arrays encoded as json
        result = self.simulation_environment.render(self.simulation_id)
        obs_data = json.loads(result)
        obs_data_dict = dict(obs_data)
        if self.render_mode == "human":
            for key in obs_data:
                print(f"{key} -> {obs_data_dict[key]}")
            return
        elif self.render_mode == "ansi":
            return str(obs_data)
        else:
            return super(SingleDC, self).render()

    def close(self):
        # close the resources
        self.gateway.close()

    def _raw_info_to_dict(self, raw_info):
        info = {
            "job_wait_reward": raw_info.getJobWaitReward(),
            "running_vm_cores_reward": raw_info.getRunningVmCoresReward(),
            "unutilized_vm_cores_reward": raw_info.getUnutilizedVmCoresReward(),
            "invalid_reward": raw_info.getInvalidReward(),
            "isValid": raw_info.isValid(),
            "host_metrics": json.loads(raw_info.getHostMetricsAsJson()),
            "vm_metrics": json.loads(raw_info.getVmMetricsAsJson()),
            "job_metrics": json.loads(raw_info.getJobMetricsAsJson()),
            "job_wait_time": json.loads(raw_info.getJobWaitTimeAsJson()),
            "unutilized_vm_core_ratio": raw_info.getUnutilizedVmCoreRatio(),
            "dotString": raw_info.getDotString(),
        }
        return info

    def _to_nparray(self, raw_obs):
        obs = list(raw_obs)
        return np.array(obs, dtype=np.float32)
