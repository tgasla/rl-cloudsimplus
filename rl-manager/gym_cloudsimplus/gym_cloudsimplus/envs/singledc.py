import gymnasium as gym
import os
import json
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters
import numpy as np

# TODO: the two environments should inherit a BaseEnvironment class
# so I do not have to repeat the same code in both of them
# TODO: the two environments should support both cntinuous and
# discrete action spaces

"""
Action space

A vector of 2 continuous numbers

The first number is in range [-1,1]
Positive value - create a VM
The range shows the id of host that the VM will be placed

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


# Based on https://gymnasium.farama.org/api/env/
class SingleDC(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}
    address = os.getenv("CLOUDSIM_GATEWAY_HOST", "gateway")
    port = os.getenv("CLOUDSIM_GATEWAY_PORT", "25333")
    parameters = GatewayParameters(
        address=address,
        port=int(port),
        auto_convert=True
    )

    def __init__(
        self,
        split_large_jobs = "true",
        max_job_pes = "1",
        host_pe_mips = "10",
        host_pes = "10",
        datacenter_hosts_cnt = "10",
        reward_job_wait_coef = "0.3",
        reward_util_coef = "0.3",
        reward_invalid_coef = "0.4",
        jobs_as_json = None,
        jobs_from_file = None,
        simulation_speedup = "1",
        render_mode = None,
        job_log_dir = None,
        max_steps = "5000"
    ):
  
        super().__init__()

        self.gateway = JavaGateway(gateway_parameters=self.parameters)
        self.simulation_environment = self.gateway.entry_point

        self.episode_details = None
        self.host_metrics = None
        self.vm_metrics = None
        self.job_metrics = None
        self.job_wait_time = None
        self.unutilized_active = None
        self.unutilized_all = None
        self.step_counter = None
        self.max_steps = int(max_steps)

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5,),
            dtype=np.float32
        )

        # These parameters are passed when calling gym.make in learn.py
        # to the java cloudsimplus gateway.
        # If a parameter is not defined, it gets a default value.
        params = {
            "SPLIT_LARGE_JOBS": split_large_jobs,
            "HOST_PE_CNT": host_pes,
            "HOST_PE_MIPS": host_pe_mips,
            "DATACENTER_HOSTS_CNT" :datacenter_hosts_cnt,
            "REWARD_JOB_WAIT_COEF": reward_job_wait_coef,
            "REWARD_UTILIZATION_COEF": reward_util_coef,
            "REWARD_INVALID_COEF": reward_invalid_coef,
            "MAX_JOB_PES": max_job_pes,
            "SIMULATION_SPEEDUP": simulation_speedup,
            "JOB_LOG_DIR": job_log_dir,
            "MAX_STEPS": max_steps
        }

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                "Invalid render mode"
                'Render modes allowed: ["human" | "ansi"]'
            )
    
        self.render_mode = render_mode

        if jobs_as_json is not None:
            params["SOURCE_OF_JOBS"] = "PARAMS"
            params["JOBS"] = jobs_as_json
        elif "jobs_from_file" is not None:
            params["JOBS_FILE"] = jobs_from_file

        self.simulation_id = self.simulation_environment.createSimulation(params)

    def reset(self, seed=None, options=None):
        super().reset()

        self.episode_details = {
            "state": [],
            "action": [],
            "reward": [],
            "job_wait_reward": [],
            "util_reward": [],
            "invalid_reward": [],
            "next_state": []
        }

        self.host_metrics = []
        self.vm_metrics = []
        self.job_metrics = []
        self.job_wait_time = []
        self.unutilized_active = []
        self.unutilized_all = []

        self.step_counter = 0

        result = self.simulation_environment.reset(self.simulation_id)
        self.simulation_environment.seed(self.simulation_id)

        raw_obs = result.getObs()
        obs = self._to_nparray(raw_obs)
        raw_info = result.getInfo()
        info = self._raw_info_to_dict(raw_info)
        
        self.episode_details["state"].append(list(raw_obs))

        # self.host_metrics.append(list(info["host_metrics"]))
        # self.vm_metrics.append(list(info["vm_metrics"]))
        # self.job_metrics.append(list(info["job_metrics"]))
        # self.job_wait_time.append(list(info["job_wait_time"]))
        # self.unutilized_active.append(info["unutilized_active"])
        # self.unutilized_all.append(info["unutilized_all"])

        return obs, info

    def step(self, action):
        # Py4J cannot translate np.arrary(dtype=np.float32) to java double[]
        # Fix1: make it dtype=np.float64 and for some reason it works :)
        # Fix2: before sending it to java, convert it to python list first
        # Here, we adopt Fix2
        action = action.tolist()
        result = self.simulation_environment.step(self.simulation_id, action)


        reward = result.getReward()
        raw_info = result.getInfo()
        terminated = result.isDone()
        truncated = False
        raw_obs = result.getObs()
        obs = self._to_nparray(raw_obs)

        info = self._raw_info_to_dict(raw_info)

        self.episode_details["action"].append(action)
        self.episode_details["reward"].append(reward)
        self.episode_details["job_wait_reward"].append(info["job_wait_reward"])
        self.episode_details["util_reward"].append(info["util_reward"])
        self.episode_details["invalid_reward"].append(info["invalid_reward"])
        self.episode_details["next_state"].append(list(raw_obs))

        self.episode_details["state"].append(list(raw_obs))

        self.host_metrics.append(list(info["host_metrics"]))
        self.vm_metrics.append(list(info["vm_metrics"]))
        self.job_metrics.append(list(info["job_metrics"]))

        # TODO: NEEDS FIX, FOR SOME REASON IT IS ALWAYS 0.0 OR 1.0
        if len(list(info["job_wait_time"])) > 0:
            self.job_wait_time.append(list(info["job_wait_time"]))
        if info["unutilized_active"] != -1.0:
            self.unutilized_active.append(info["unutilized_active"])
        if info["unutilized_all"] != -1.0:
            self.unutilized_all.append(info["unutilized_all"])

        if self.render_mode == "human":
            self.render()

        self.step_counter += 1
        # limit the maximum allowed number of timesteps in an episode
        if self.step_counter >= self.max_steps:
            truncated = True

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

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
            return super().render()
    
    def close(self):
        # close the resources
        self.gateway.close()
    
    def _raw_info_to_dict(self, raw_info):
        info = {
            "job_wait_reward": raw_info.getJobWaitReward(),
            "util_reward": raw_info.getUtilReward(),
            "invalid_reward": raw_info.getInvalidReward(),
            "ep_job_wait_rew_mean": raw_info.getEpJobWaitRewardMean(),
            "ep_util_rew_mean": raw_info.getEpUtilRewardMean(),
            "ep_valid_count": raw_info.getEpValidCount(),
            "host_metrics": json.loads(raw_info.getHostMetricsAsJson()),
            "vm_metrics": json.loads(raw_info.getVmMetricsAsJson()),
            "job_metrics": json.loads(raw_info.getJobMetricsAsJson()),
            "job_wait_time": json.loads(raw_info.getJobWaitTimeAsJson()),
            "unutilized_active": raw_info.getUnutilizedActive(),
            "unutilized_all": raw_info.getUnutilizedAll()
        }
        return info

    def _to_nparray(self, raw_obs):
        obs = list(raw_obs)
        return np.array(obs, dtype=np.float32)
    
    def _flatten(self, test_list):
        if isinstance(test_list, list):
            temp = []
            for ele in test_list:
                temp.extend(self._flatten(ele))
            return temp
        else:
            return [test_list]