import gymnasium as gym
import gym_cloudsimplus
import json
import time
import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
import dummy_agents
import torch
import argparse
from read_swf import SWFReader

def human_format(num):
    num = float(f"{num:.3f}")
    magnitude = 0
    suffix = ['', 'K', 'M', 'B', 'T']
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    return f"{num:.0f}" + suffix[magnitude]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

swf_reader = SWFReader()
jobs = swf_reader.read("mnt/LLNL-Atlas-2006-2.1-cln.swf", jobs_to_read=10)

env = gym.make(
    "SmallDC-v0",
    # jobs_as_json=json.dumps(jobs),
    simulation_speedup="10000",
    split_large_jobs="true",
    render_mode="ansi"
)

it = 0
reward_sum = 0

parser = argparse.ArgumentParser()
parser.add_argument("algorithm", 
                    type=str,
                    choices=["DQN", "A2C", "PPO", 
                             "RNG", "DDPG", "HER", 
                             "SAC", "TD3"
                             ],
                    help="The RL algorithm to train")
parser.add_argument("timesteps", type=int,
                    help="The number of timesteps to train")
args = parser.parse_args()
algorithm_str = str(args.algorithm).upper()
timesteps = int(args.timesteps)

rng_algorithm = False
if algorithm_str == "RNG":
    rng_algorithm = True

# Not needed because we have the choices parameter in add_argument
# if not rng_algorithm and not hasattr(sb3, algorithm_str):
#     raise NameError(f"RL algorithm {algorithm_str} was not found")

tb_log = f"./tb-logs/{algorithm_str}/"

if rng_algorithm:
    algorithm = getattr(dummy_agents, algorithm_str)
    policy = "RngPolicy"
else:
    algorithm = getattr(sb3, algorithm_str)
    policy = "MlpPolicy"

model = algorithm(
    policy=policy,
    env=env,
    verbose=True,
    tensorboard_log=tb_log,
    device=device
)

# Model training
model.learn(
    total_timesteps=timesteps,
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"{algorithm_str}_{human_format(timesteps)}"
)

# Model evaluation
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10,
    render = True
)

print(f"Mean Reward: {mean_reward} +/- {std_reward}")

model_path = f"./storage/{algorithm_str}_{human_format(timesteps)}_{int(time.time())}"

model.save(model_path)

del model

model =algorithm.load(model_path)

# Model deployment
obs, info = env.reset()

done = False
while not done:
    action, _states = model.predict(obs)
    print(f"ACTION = {action}")
    obs, reward, terminated, truncated, info = env.step(int(action))
    print(f"Iteration: {it}")
    print("State Space:")
    print("-" * 50)
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
    done = terminated or truncated
    if terminated:
        print(f"Episode finished! Reward sum: {reward_sum}")

env.close()