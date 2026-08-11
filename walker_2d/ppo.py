import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

"""
ENVIRONMENT
"""
def make_env(env_id="Walker2d-v5", seed=41):
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

"""
DEVICE
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
NETWORK
"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # Extract dimension sizes safely (handles single or vectorized envs)
        obs_shape = envs.single_observation_space.shape if hasattr(envs, "single_observation_space") else envs.observation_space.shape
        action_shape = envs.single_action_space.shape if hasattr(envs, "single_action_space") else envs.action_space.shape

        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))

        # Actor outputs action mean
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )
        # Parameterized standard deviation for continuous action space
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic outputs state value V(s) scalar (dim=1)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # Sum probabilities across the action dimensions
        return action, probs.log_prob(action).sum(axis=-1), probs.entropy().sum(axis=-1), self.critic(x)

"""
PPO ROLLOUT BUFFER
"""
class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        self.returns = self.advantages + self.values