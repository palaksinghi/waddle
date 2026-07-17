"""" this is the first file used for to check the env, sample of observation space,action space and gym installation"""

import gymnasium as gym

class PusherEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.env = gym.make("Pusher-v5", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()