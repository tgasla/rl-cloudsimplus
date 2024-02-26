import gymnasium as gym
import gym_cloudsimplus
import json
import sys
from py4j.java_gateway import JavaGateway
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import argparse
from read_swf import SWFReader

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

swf_reader = SWFReader()
jobs = swf_reader.read("mnt/LLNL-Atlas-2006-2.1-cln.swf", lines_to_read=1000)

gateway = JavaGateway()
simulation_environment = gateway.entry_point

env = gym.make(
    "SingleDCAppEnv-v0",
    #initial_vm_count=initial_vm_count,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup="10000",
    split_large_jobs="true",
    render_mode="human"
)

it = 0
reward_sum = 0

parser = argparse.ArgumentParser()
parser.add_argument("algorithm", type=str,
                    help="The RL algorithm to train")
parser.add_argument("timesteps", type=int,
                    help="The number of timesteps to train")
args = parser.parse_args()
algorithm_str = str(args.algorithm).upper()
timesteps = int(args.timesteps)

if not hasattr(sb3, algorithm_str):
    raise NameError(f"RL algorithm {algorithm_str} not found")

tb_log = f"./tb_logs/{algorithm_str}/"

algorithm = getattr(sb3, algorithm_str)

model = algorithm(
    "MlpPolicy",
    env,
    verbose=True,
    tensorboard_log=tb_log,
    device=device)

model.learn(
    total_timesteps=timesteps,
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"{algorithm_str}_{human_format(timesteps)}"
)

mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10,
    render = True
)
print("Mean Reward: {} +/- {}".format(mean_reward, std_reward))

obs, info = env.reset()

while True:
    action, _states = model.predict(obs)
    print(f"ACTION = {action}")
    obs, reward, terminated, truncated, info = env.step(int(action))
    print(f"Iteration: {it}")
    print("State Space:")
    print("-" * 20)
    print(f"avgCPUUtilizationHistory: {obs[0]}")
    print(f"vmAllocatedRatioHistory: {obs[1]}")
    print(f"p90CPUUtilizationHistory: {obs[2]}")
    print(f"avgMemoryUtilizationHistory: {obs[3]}")
    print(f"p90MemoryUtilizationHistory: {obs[4]}")
    print(f"waitingJobsRatioGlobalHistory: {obs[5]}")
    print(f"waitingJobsRatioRecentHistory: {obs[6]}")
    print(f"Current Reward: {reward}")
    reward_sum += reward

    it += 1
    if terminated or truncated:
        print(f"Episode finished! Reward sum: {reward_sum}")
        break
