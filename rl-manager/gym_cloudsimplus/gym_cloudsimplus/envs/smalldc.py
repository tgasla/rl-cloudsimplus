import gymnasium as gym
import os
import json
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters
import numpy as np

# Action space - 7 discrete actions
# ACTION_NOTHING: 0
# ACTION_ADD_VM: 1
# ACTION_REMOVE_VM: 2
# ACTION_ADD_MEDIUM_VM: 3
# ACTION_REMOVE_MEDIUM_VM: 4
# ACTION_ADD_LARGE_VM: 5
# ACTION_REMOVE_LARGE_VM: 6

# Observation space - 7 continuous values within [0-1]
# "vmAllocatedRatioHistory",
# "avgCPUUtilizationHistory",
# "p90CPUUtilizationHistory",
# "avgMemoryUtilizationHistory",
# "p90MemoryUtilizationHistory",
# "waitingJobsRatioGlobalHistory",
# "waitingJobsRatioRecentHistory"

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
class SmallDC(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, **kwargs):
        super().__init__()
        # actions are identified by integers 0-n
        self.num_of_actions = 7
        self.action_space = spaces.Discrete(self.num_of_actions)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

        # Get parameters when calling gym.make on learn.py
        #   if a parameter is not defined, it gets a default value
        params = {
            'INITIAL_L_VM_COUNT':
                kwargs.get('initial_l_vm_count', '1'),
            'INITIAL_M_VM_COUNT':
                kwargs.get('initial_m_vm_count', '1'),
            'INITIAL_S_VM_COUNT':
                kwargs.get('initial_s_vm_count', '1'),
            'SIMULATION_SPEEDUP':
                kwargs.get('simulation_speedup', '1.0'),
            'SPLIT_LARGE_JOBS':
                kwargs.get('split_large_jobs', 'false'),
            'QUEUE_WAIT_PENALTY':
                kwargs.get('queue_wait_penalty', '0.00001'),
            'VM_RUNNING_HOURLY_COST':
                kwargs.get('vm_running_hourly_cost', '0.2'),
            'HOST_S_PE_CNT':
                kwargs.get('host_s_pe_cnt', '22'),
            'HOST_S_PE_MIPS':
                kwargs.get('host_s_pe_mips', '110000'),
            'HOST_S_RAM':
                kwargs.get('host_s_ram', '128000'),
            'HOST_M_PE_CNT':
                kwargs.get('host_m_pe_cnt', '18'),
            'HOST_M_PE_MIPS':
                kwargs.get('host_m_pe_mips', '98000'),
            'HOST_M_RAM':
                kwargs.get('host_m_ram', '256000'),
            'HOST_L_PE_CNT':
                kwargs.get('host_l_pe_cnt', '10'),
            'HOST_L_PE_MIPS':
                kwargs.get('host_l_pe_mips', '40000'),
            'HOST_L_RAM':
                kwargs.get('host_l_ram', '384000'),
            'HOST_XL_PE_CNT':
                kwargs.get('host_xl_pe_cnt', '6'),
            'HOST_XL_PE_MIPS':
                kwargs.get('host_xl_pe_mips', '22500'),
            'HOST_XL_RAM':
                kwargs.get('host_xl_ram', '786000'),
            'HOST_2XL_PE_CNT':
                kwargs.get('host_2xl_pe_cnt', '6'),
            'HOST_2XL_PE_MIPS':
                kwargs.get('host_2xl_pe_mips', '29000'),
            'HOST_2XL_RAM':
                kwargs.get('host_2xl_ram', '1500000'),
            'HOST_BW':
                kwargs.get('host_bw', '40000'),
            'HOST_SIZE':
                kwargs.get('hostSize', '50000'),
            'QUEUE_WAIT_PENALTY':
                kwargs.get('queue_wait_penalty', '0.00001'),
            'DATACENTER_S_HOSTS_CNT':
                kwargs.get('datacenter_s_hosts_cnt', '20'),
            'DATACENTER_M_HOSTS_CNT':
                kwargs.get('datacenter_m_hosts_cnt', '5'),
            'DATACENTER_L_HOSTS_CNT':
                kwargs.get('datacenter_l_hosts_cnt', '20'),
            'DATACENTER_XL_HOSTS_CNT':
                kwargs.get('datacenter_xl_hosts_cnt', '3'),
            'DATACENTER_2XL_HOSTS_CNT':
                kwargs.get('datacenter_2xl_hosts_cnt', '2'),
            'BASIC_VM_RAM':
                kwargs.get('basic_vm_ram', '8192'),
            'BASIC_VM_PE_CNT':
                kwargs.get('basic_vm_pe_count','2'),
            'VM_SHUTDOWN_DELAY':
                kwargs.get('vm_shutdown_delay', '0'),
            'MAX_VMS_PER_SIZE':
                kwargs.get('max_vms_Per_size', '50'),
            'PRINT_JOBS_PERIODICALLY':
                kwargs.get('print_jobs_periodically', 'false'),
            'PAYING_FOR_THE_FULL_HOUR':
                kwargs.get('paying_for_the_full_hour', 'false'),
            'STORE_CREATED_CLOUDLETS_DATACENTER_BROKER':
                kwargs.get('storeCreatedCloudletsDatacenterBroker', 'false')
        }

        self.render_mode = kwargs.get('render_mode', 'None')

        if 'jobs_as_json' in kwargs:
            params['SOURCE_OF_JOBS'] = 'PARAMS'
            params['JOBS'] = kwargs['jobs_as_json']
        elif 'jobs_from_file' in kwargs:
            params['JOBS_FILE'] = kwargs['jobs_from_file']

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
        raw_obs = simulation_environment.reset(self.simulation_id)
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
            return str(obs_data)
        else:
            return super().render()

    def close(self):
        # close the resources
        simulation_environment.close(self.simulation_id)

    def seed(self):
        simulation_environment.seed(self.simulation_id)
