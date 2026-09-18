
import gymnasium as gym
env=gym.make("Humanoid-v5",render_mode="human")
obs,info=env.reset()
for episodes in range(1000):
    action=print(env.action_space.sample)
    action=env.action_space.sample()
    observation=print(env.observation_space.sample)
    obs,reward,terminated,truncated,info=env.step(action)
    obs,info=env.reset()
env.close()