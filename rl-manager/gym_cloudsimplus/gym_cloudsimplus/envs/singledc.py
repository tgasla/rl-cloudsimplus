from operator import is_
import gymnasium as gym
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
    # default port = 25333
    parameters = GatewayParameters(address="gateway", auto_convert=True)

    def __init__(
        self,
        params,
        jobs_as_json="[]",
        render_mode="ansi",
    ):
        super(SingleDC, self).__init__()

        self.gateway = JavaGateway(gateway_parameters=self.parameters)
        self.simulation_environment = self.gateway.entry_point
        self.state_as_tree_array = params["state_as_tree_array"]

        # host_count = params["host_count"]
        host_pes = params["host_pes"]
        small_vm_pes = params["small_vm_pes"]

        # if you want to support 1-10 hosts then when calculating max_vms_count and
        # observation rows, put self.max_hosts instead of host_count
        # and in action_space put self.max_hosts instead of host_count

        self.action_types_count = 3
        self.max_hosts = 10
        self.types_of_vms_count = 3

        # it makes sense to assume that the minimum amount of cores per job will be 1
        self.min_job_pes = 1
        self.max_vms = self.max_hosts * int(host_pes) // int(small_vm_pes)
        self.max_jobs = self.max_vms * int(small_vm_pes) // self.min_job_pes

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
            np.array(
                [
                    self.action_types_count,
                    self.max_hosts,
                    self.max_vms,
                    self.types_of_vms_count,
                ]
            )
        )

        if self.state_as_tree_array:
            self.observation_length = 1 + self.max_hosts + self.max_vms + self.max_jobs
            self.max_cores_per_node = 101
            self.observation_space = spaces.MultiDiscrete(
                self.max_cores_per_node * np.ones(self.observation_length)
            )
        else:
            self.observation_rows = 1 + self.max_hosts + self.max_vms + self.max_jobs
            self.observation_cols = 4
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

        self.simulation_id = self.simulation_environment.createSimulation(
            params, jobs_as_json
        )

    def reset(self, seed=None, options=None):
        super(SingleDC, self).reset()
        if seed is None:
            seed = 0
        result = self.simulation_environment.reset(self.simulation_id, seed)

        if self.state_as_tree_array:
            raw_obs = result.getObservationTreeArray()
        else:
            raw_obs = result.getObservationMatrix()

        obs = self._to_nparray(raw_obs)
        padded_obs = np.resize(obs, self.observation_length)
        padded_obs[len(obs) :] = 0

        print(padded_obs)

        raw_info = result.getInfo()
        info = self._raw_info_to_dict(raw_info)

        return padded_obs, info

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

        if self.state_as_tree_array:
            raw_obs = result.getObservationTreeArray()
        else:
            raw_obs = result.getObservationMatrix()

        obs = self._to_nparray(raw_obs)
        padded_obs = np.resize(obs, self.observation_length)
        padded_obs[len(obs) :] = 0

        info = self._raw_info_to_dict(raw_info)

        if self.render_mode == "human":
            self.render()

        return (padded_obs, reward, terminated, truncated, info)

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
        self.simulation_environment.close(self.simulation_id)
        # close the python client side
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
            "observation_tree_array": json.loads(
                raw_info.getObservationTreeArrayAsJson()
            ),
            # "dot_string": raw_info.getDotString(),
        }
        return info

    def _to_nparray(self, raw_obs):
        obs = list(raw_obs)
        return np.array(obs, dtype=np.float32)
