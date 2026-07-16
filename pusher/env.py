"""" this is the first file used for to check the env, sample of observation space,action space and gym installation"""

import gymnasium as gym

env = gym.make("Pusher-v5",render_mode="human")

obs, info = env.reset()
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
# env.close()