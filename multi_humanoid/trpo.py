"""
TRPO (Trust Region Policy Optimization) - core algorithm.

Implements:
  - Gaussian MLP policy (diagonal covariance)
  - MLP value function baseline
  - Generalized Advantage Estimation (GAE-lambda)
  - Conjugate gradient solver for  F x = g
  - Fisher-vector product via KL Hessian-vector product trick
  - Backtracking line search enforcing the KL trust-region + surrogate improvement

Reference: Schulman et al., "Trust Region Policy Optimization" (2015).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            dist = self.forward(obs)
            if deterministic:
                return dist.mean
            return dist.sample()

    def log_prob(self, obs, act):
        dist = self.forward(obs)
        return dist.log_prob(act).sum(-1)

    def kl(self, obs, old_mu, old_std):
        dist = self.forward(obs)
        mu, std = dist.mean, dist.stddev
        var, old_var = std.pow(2), old_std.pow(2)
        kl = (
            torch.log(std / old_std)
            + (old_var + (old_mu - mu).pow(2)) / (2.0 * var)
            - 0.5
        )
        return kl.sum(-1).mean()

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].view_as(p))
            idx += n


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        self.v_net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)

""""
hyperparamter
"""


GAMMA=0.9995
LAMDA=0.998
VF_ITER=5
N_ITERS=10
VF_LR=0.5e-3
MAX_KL=0.01
DAMPING=0.1
BACKTRACK_COEFF=0.5
BACKTRACK_ITERS=10
CG_ITERS=10

"""
GAE(GENERAL ADVANTAGE ESTIMATION)
"""


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAMDA):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    values_ext = np.append(values, last_value)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def conjugate_gradient(fvp_fn, b, n_iters=N_ITERS, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(n_iters):
        fvp = fvp_fn(p)
        alpha = rdotr / (torch.dot(p, fvp) + 1e-8)
        x += alpha * p
        r -= alpha * fvp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x


def flat_grad(y, params, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        y,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    grads = [
        torch.zeros_like(p) if g is None else g
        for p, g in zip(params, grads)
    ]
    return torch.cat([g.reshape(-1) for g in grads])


class TRPO:
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), max_kl=MAX_KL,
                 cg_iters=CG_ITERS, damping=DAMPING, backtrack_iters=BACKTRACK_ITERS, backtrack_coeff=BACKTRACK_COEFF,
                 vf_lr=VF_LR, vf_iters=VF_ITER, gamma=GAMMA, lam=LAMDA):
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden).to(device)
        self.value = ValueNet(obs_dim, hidden).to(device)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=vf_lr)

        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.damping = damping
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.vf_iters = vf_iters
        self.gamma = gamma
        self.lam = lam

    def _surrogate_loss(self, obs, act, adv, old_logp):
        logp = self.policy.log_prob(obs, act)
        ratio = torch.exp(logp - old_logp)
        return (ratio * adv).mean()

    def update(self, obs, act, adv, ret, old_logp, old_mu, old_std):
        obs, act = obs.to(device), act.to(device)
        adv, ret = adv.to(device), ret.to(device)
        old_logp = old_logp.to(device)
        old_mu, old_std = old_mu.to(device), old_std.to(device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        params = list(self.policy.parameters())

        loss = self._surrogate_loss(obs, act, adv, old_logp)
        g = flat_grad(loss, params, retain_graph=False, create_graph=False)

        def fvp(v):
            kl = self.policy.kl(obs, old_mu, old_std)

            kl_grads = torch.autograd.grad(
                kl,
                params,
                create_graph=True,
                retain_graph=True,
            )
            flat_kl_grad = torch.cat([kg.reshape(-1) for kg in kl_grads])

            grad_v = torch.dot(flat_kl_grad, v)

            hvps = torch.autograd.grad(
                grad_v,
                params,
                retain_graph=False,
            )
            hvp = torch.cat([hg.reshape(-1) for hg in hvps])

            return hvp + self.damping * v

        step_dir = conjugate_gradient(fvp, g, n_iters=self.cg_iters)
        shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
        step_size = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_size * step_dir

        expected_improve = torch.dot(g, full_step)
        old_params = self.policy.get_flat_params()
        old_loss = loss.item()

        success = False
        kl = 0.0
        for i in range(self.backtrack_iters):
            coeff = self.backtrack_coeff ** i
            new_params = old_params + coeff * full_step
            self.policy.set_flat_params(new_params)

            with torch.no_grad():
                new_loss = self._surrogate_loss(obs, act, adv, old_logp).item()
                kl = self.policy.kl(obs, old_mu, old_std).item()

            improve = new_loss - old_loss
            expected = expected_improve.item() * coeff
            ratio = improve / expected if abs(expected) > 1e-8 else 0.0

            if kl <= self.max_kl and improve > 0 and ratio > 0.1:
                success = True
                break

        if not success:
            self.policy.set_flat_params(old_params)

        v_loss = torch.tensor(0.0)
        for _ in range(self.vf_iters):
            self.value_optim.zero_grad()
            v_pred = self.value(obs)
            v_loss = ((v_pred - ret) ** 2).mean()
            v_loss.backward()
            self.value_optim.step()

        return {
            "surrogate_loss": old_loss,
            "kl": kl if success else 0.0,
            "value_loss": v_loss.item(),
            "step_accepted": success,
        }