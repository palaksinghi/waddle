import gymnasium as gym 

class HumanoidEnv(gym.Env):
    def __init__ (self, render_mode=None):
     self.env=gym.make("humanoid-v5",render_mode=render_mode)
     self.observation_space=self.env.observation_space
     self.action_space=self.env.action_space
     
    def reset(self,seed=None,options=None):
        return self.env.reset(seed=seed,options=options)

    def step(self,action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()