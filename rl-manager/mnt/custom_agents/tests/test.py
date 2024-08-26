import gymnasium as gym
from dummy_agents import RNG

# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Instantiate the agent
model = RNG("MlpPolicy", env, verbose=True)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e2), progress_bar=False)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
