import gymnasium as gym
import os
import json
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters
import numpy as np

# Available actions
# ACTION_NOTHING: 0
# ACTION_ADD_VM: 1
# ACTION_REMOVE_VM: 2
# ACTION_ADD_MEDIUM_VM: 3
# ACTION_REMOVE_MEDIUM_VM: 4
# ACTION_ADD_LARGE_VM: 5
# ACTION_REMOVE_LARGE_VM: 6

address = os.getenv('CLOUDSIM_GATEWAY_HOST', 'gateway')
port = os.getenv('CLOUDSIM_GATEWAY_PORT', '25333')

parameters = GatewayParameters(address=address,
                               port=int(port),
                               auto_convert=True)
gateway = JavaGateway(gateway_parameters=parameters)
simulation_environment = gateway.entry_point


def to_string(java_array):
    return gateway.jvm.java.util.Arrays.toString(java_array)


def to_nparray(raw_obs):
    obs = list(raw_obs)
    return np.array(obs, dtype=np.float32)


# Based on https://gymnasium.farama.org/api/env/
class SingleDCAppEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, **kwargs):
        super().__init__()
        # actions are identified by integers 0-n
        self.num_of_actions = 7
        self.action_space = spaces.Discrete(self.num_of_actions)

        # observation metrics - all within 0-1 range
        # "vmAllocatedRatioHistory",
        # "avgCPUUtilizationHistory",
        # "p90CPUUtilizationHistory",
        # "avgMemoryUtilizationHistory",
        # "p90MemoryUtilizationHistory",
        # "waitingJobsRatioGlobalHistory",
        # "waitingJobsRatioRecentHistory"
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.float32
        )
        # mandatory args
        params = {
            'INITIAL_VM_COUNT': kwargs.get('initial_vm_count', '1'),
            'SOURCE_OF_JOBS': 'PARAMS',
            'JOBS': kwargs.get('jobs_as_json', '[]'),
            'SIMULATION_SPEEDUP': kwargs.get('simulation_speedup', '1.0'),
            'SPLIT_LARGE_JOBS': kwargs.get('split_large_jobs', 'false'),
        }

        self.render_mode = kwargs.get('render_mode', 'None')

        # optional arg
        if 'queue_wait_penalty' in kwargs:
            params['QUEUE_WAIT_PENALTY'] = kwargs['queue_wait_penalty']

        self.simulation_id = simulation_environment.createSimulation(params)

    def step(self, action):
        if type(action) == np.int64:
            action = action.item()
        result = simulation_environment.step(self.simulation_id, action)
        reward = result.getReward()
        terminated = result.isDone()
        truncated = False
        raw_obs = result.getObs()

        obs = to_nparray(raw_obs)

        if self.render_mode == 'human':
            self.render()
            print(f"Current reward is {reward}")

        return (
            obs,
            reward,
            terminated,
            truncated,
            {}
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        result = simulation_environment.reset(self.simulation_id)
        raw_obs = result.getObs()
        obs = to_nparray(raw_obs)
        info = {}
        return obs, info

    def render(self):
        # result is a string with arrays encoded as json
        result = simulation_environment.render(self.simulation_id)
        obs_data = json.loads(result)
        if self.render_mode == 'human':
            print("Observation metrics:")
            print("-" * 40)
            print(f"avgCPUUtilizationHistory: {obs_data[0]}")
            print(f"vmAllocatedRatioHistory: {obs_data[1]}")
            print(f"p90CPUUtilizationHistory: {obs_data[2]}")
            print(f"avgMemoryUtilizationHistory: {obs_data[3]}")
            print(f"p90MemoryUtilizationHistory: {obs_data[4]}")
            print(f"waitingJobsRatioGlobalHistory: {obs_data[5]}")
            print(f"waitingJobsRatioRecentHistory: {obs_data[6]}")
            return
        elif self.render_mode == 'ansi':
            return obs_data
        else:
            return super().render()

    def close(self):
        # close the resources
        simulation_environment.close(self.simulation_id)

    def seed(self):
        simulation_environment.seed(self.simulation_id)