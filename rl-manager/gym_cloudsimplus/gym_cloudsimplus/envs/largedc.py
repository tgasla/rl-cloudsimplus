import gymnasium as gym
import os
import json
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters
import numpy as np

# TODO: the two environments should inherit a BaseEnvironment class
# so I do not have to repeat the same code in both of them
"""
Action space

A vector of 2 continuous numbers

The first number is in range [-1,1]
Positive value - create a VM
Negative value - destroy a VM
The range shows the id of the VM that will be destroyed

The second number is in range [0,1]
It shows the size of the VM that will be created/destroyed



Observation space

A vector of 7 continuous values 

All values are within range [0,1]

"vmAllocatedRatioHistory",
"avgCPUUtilizationHistory",
"p90CPUUtilizationHistory",
"avgMemoryUtilizationHistory",
"p90MemoryUtilizationHistory",
"waitingJobsRatioGlobalHistory",
"waitingJobsRatioRecentHistory"
"""

address = os.getenv("CLOUDSIM_GATEWAY_HOST", "gateway")
port = os.getenv("CLOUDSIM_GATEWAY_PORT", "25333")

parameters = GatewayParameters(
    address=address,
    port=int(port),
    auto_convert=True
)
gateway = JavaGateway(gateway_parameters=parameters)
simulation_environment = gateway.entry_point


def to_string(java_array):
    return gateway.jvm.java.util.Arrays.toString(java_array)


def to_nparray(raw_obs):
    obs = list(raw_obs)
    return np.array(obs, dtype=np.float32)


# Based on https://gymnasium.farama.org/api/env/
class LargeDC(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, **kwargs):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]),
            shape=(2,),
            # py4j translates this tuple of doubles into ArrayList<>
            dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

        # These parameters are passed when calling gym.make in learn.py
        # to the java cloudsimplus gateway.
        # If a parameter is not defined, it gets a default value.
        params = {
            "INITIAL_L_VM_COUNT":
                kwargs.get("initial_l_vm_count", "1"),
            "INITIAL_M_VM_COUNT":
                kwargs.get("initial_m_vm_count", "1"),
            "INITIAL_S_VM_COUNT":
                kwargs.get("initial_s_vm_count", "1"),
            "SIMULATION_SPEEDUP":
                kwargs.get("simulation_speedup", "1.0"),
            "SPLIT_LARGE_JOBS":
                kwargs.get("split_large_jobs", "false"),
            "QUEUE_WAIT_PENALTY":
                kwargs.get("queue_wait_penalty", "0.00001"),
            "VM_RUNNING_HOURLY_COST":
                kwargs.get("vm_running_hourly_cost", "0.2"),
            "HOST_PE_MIPS":
                kwargs.get("host_pe_mips", "222122"),
            "HOST_BW":
                kwargs.get("host_bw", "100000"),
            # 192 GB of RAM = 192 * 1024 MB
            "HOST_RAM":
                kwargs.get("host_ram", "196608"),
            "HOST_SIZE":
                kwargs.get("hostSize", "400000"),
            "HOST_PE_CNT":
                kwargs.get("hostPeCnt", "40"),
            "DATACENTER_HOSTS_CNT":
                kwargs.get("datacenter_hosts_cnt", "500"),
            "BASIC_VM_RAM":
                kwargs.get("basic_vm_ram", "8192"),
            "BASIC_VM_PE_CNT":
                kwargs.get("basic_vm_pe_count","2"),
            "VM_SHUTDOWN_DELAY":
                kwargs.get("vm_shutdown_delay", "0.0"),
            "MAX_VMS_PER_SIZE":
                kwargs.get("max_vms_per_size", "500"),
            "PRINT_JOBS_PERIODICALLY":
                kwargs.get("print_jobs_periodically", "false"),
            "PAYING_FOR_THE_FULL_HOUR":
                kwargs.get("paying_for_the_full_hour", "false"),
            "STORE_CREATED_CLOUDLETS_DATACENTER_BROKER":
                kwargs.get("storeCreatedCloudletsDatacenterBroker", "false")
        }

        render_mode = kwargs.get("render_mode", None)
        assert render_mode is None \
            or render_mode in self.metadata["render_modes"]
    
        self.render_mode = render_mode

        if "jobs_as_json" in kwargs:
            params["SOURCE_OF_JOBS"] = "PARAMS"
            params["JOBS"] = kwargs["jobs_as_json"]
        elif "jobs_from_file" in kwargs:
            params["JOBS_FILE"] = kwargs["jobs_from_file"]

        self.simulation_id = simulation_environment.createSimulation(params)

    def step(self, action):
        result = simulation_environment.step(self.simulation_id, action)
        reward = result.getReward()
        terminated = result.isDone()
        truncated = False
        raw_obs = result.getObs()
        obs = to_nparray(raw_obs)

        if self.render_mode == "human":
            self.render()

        return (
            obs,
            reward,
            terminated,
            truncated,
            {}
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = simulation_environment.reset(self.simulation_id)
        obs = to_nparray(raw_obs)
        info = {}
        return obs, info

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
        result = simulation_environment.render(self.simulation_id)
        obs_data = json.loads(result)
        if self.render_mode == "human":
            print("Observation state:")
            print("-" * 40)
            print(f"avgCPUUtilizationHistory: {obs_data[0]}")
            print(f"vmAllocatedRatioHistory: {obs_data[1]}")
            print(f"p90CPUUtilizationHistory: {obs_data[2]}")
            print(f"avgMemoryUtilizationHistory: {obs_data[3]}")
            print(f"p90MemoryUtilizationHistory: {obs_data[4]}")
            print(f"waitingJobsRatioGlobalHistory: {obs_data[5]}")
            print(f"waitingJobsRatioRecentHistory: {obs_data[6]}")
            print("-" * 40)
            return
        elif self.render_mode == "ansi":
            return str(obs_data)
        else:
            return super().render()

    def close(self):
        # close the resources
        simulation_environment.close(self.simulation_id)

    def seed(self):
        simulation_environment.seed(self.simulation_id)
