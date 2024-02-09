import gym
import gym_cloudsimplus
import json
import sys
from py4j.java_gateway import JavaGateway
from stable_baselines3 import PPO, A2C
import torch
from read_swf import jobs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

gateway = JavaGateway()
simulation_environment = gateway.entry_point

if len(sys.argv) > 1:
    initial_vm_count = sys.argv[1]
else:
    initial_vm_count = '10'

env = gym.make('SingleDCAppEnv-v0',
               initial_vm_count=initial_vm_count,
               jobs_as_json=json.dumps(jobs),
               simulation_speedup="10000",
               split_large_jobs="true",
               render_mode="human"
               )

it = 0
reward_sum = 0
model = A2C("MlpPolicy", env, device="cuda")
model.learn(total_timesteps=100_000, progress_bar=True)
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    print(f"ACTION = {action}")
    obs, reward, done, info = env.step(int(action))
    print(f'{it}, {[str(i) for i in obs]}, {reward}')
    reward_sum += reward

    it += 1
    if done:
        print(f"Episode finished! Reward sum: {reward_sum}")
        break