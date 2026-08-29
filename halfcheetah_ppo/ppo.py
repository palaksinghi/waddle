"""PPO agent and running-statistics utilities used by train.py."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64); self.var = np.ones(shape, np.float64); self.count = epsilon
    def update(self, x):
        x = np.asarray(x, np.float64); batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean; total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        self.count = total


class ObservationNormalizer:
    def __init__(self, shape, clip=10.0): self.rms, self.clip = RunningMeanStd(shape), clip
    def update(self, obs): self.rms.update(obs)
    def normalize(self, obs): return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip).astype(np.float32)


class RewardNormalizer:
    """Normalizes discounted returns, then clips each learning reward."""
    def __init__(self, num_envs, gamma=0.99, clip=10.0):
        self.returns = np.zeros(num_envs); self.rms, self.gamma, self.clip = RunningMeanStd(), gamma, clip
    def __call__(self, rewards, dones):
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        normalized = rewards / np.sqrt(self.rms.var + 1e-8)
        self.returns[dones] = 0.0
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)


class ValueNormalizer:
    def __init__(self): self.rms = RunningMeanStd()
    def update(self, targets): self.rms.update(np.asarray(targets).reshape(-1))
    def normalize(self, values): return (values - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
    def denormalize(self, values): return values * np.sqrt(self.rms.var + 1e-8) + self.rms.mean


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std); nn.init.constant_(layer.bias, bias); return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, action_dim), 0.01))
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), 1.0))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def distribution(self, obs): return Normal(self.actor(obs), self.log_std.exp().expand_as(self.actor(obs)))
    def get_action_and_value(self, obs, action=None):
        dist = self.distribution(obs)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(obs).squeeze(-1)
