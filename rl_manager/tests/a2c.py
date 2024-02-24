import gymnasium as gym
import gym_cloudsimplus
import json
import sys
from py4j.java_gateway import JavaGateway
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import torch
from read_swf import SWFReader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

swf_reader = SWFReader()
jobs = swf_reader.read("tests/LLNL-Atlas-2006-2.1-cln.swf", lines_to_read=1000)

gateway = JavaGateway()
simulation_environment = gateway.entry_point

if len(sys.argv) > 1:
    initial_vm_count = sys.argv[1]
else:
    initial_vm_count = "10"

env = gym.make(
    "SingleDCAppEnv-v0",
    initial_vm_count=initial_vm_count,
    jobs_as_json=json.dumps(jobs),
    simulation_speedup="10000",
    split_large_jobs="true",
    render_mode="human"
)

it = 0
reward_sum = 0
tb_log = "./a2c_log_cloudsimplus/"

eval_callback = EvalCallback(
    env,
    best_model_save_path='./a2c_log_cloudsimplus/',
    log_path='./a2c_log_cloudsimplus/',
    eval_freq=5000,
    render=False)

model = A2C(
    "MlpPolicy",
    env,
    verbose=True,
    tensorboard_log=tb_log,
    device=device)

model.learn(
    total_timesteps=100_000,
    progress_bar=True,
    reset_num_timesteps=False,
    callback=eval_callback,
    tb_log_name="A2C_v1"
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
