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


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Diagonal-covariance Gaussian policy for continuous action spaces."""

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

    # ---- analytic KL divergence between two diagonal Gaussians ---- #
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


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.97):
    """rewards, values, dones: 1D numpy arrays for one flattened batch of trajectories."""
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


# --------------------------------------------------------------------------- #
# Conjugate gradient
# --------------------------------------------------------------------------- #
def conjugate_gradient(fvp_fn, b, n_iters=10, residual_tol=1e-10):
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
    """Flatten autograd.grad(y, params) into a single 1D tensor.

    allow_unused=True + zero-fill keeps this safe even if some params
    don't participate in y's computation graph.
    """
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


# --------------------------------------------------------------------------- #
# TRPO update step
# --------------------------------------------------------------------------- #
class TRPO:
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), max_kl=0.01,
                 cg_iters=10, damping=0.1, backtrack_iters=10, backtrack_coeff=0.8,
                 vf_lr=1e-3, vf_iters=5, gamma=0.99, lam=0.97):
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

    # ---- surrogate loss: E[ pi(a|s)/pi_old(a|s) * A ] ---- #
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

        # ---- policy gradient of the surrogate objective (first-order only) ---- #
        loss = self._surrogate_loss(obs, act, adv, old_logp)
        g = flat_grad(loss, params, retain_graph=False, create_graph=False)

        # ---- Fisher-vector product: F v = grad( grad(KL) . v ) ---- #
        # IMPORTANT: kl must be recomputed from scratch on every call (it's
        # called ~cg_iters+1 times), and the first-order grad of kl MUST be
        # built with create_graph=True so it can itself be differentiated
        # again below. Without create_graph=True here, the second
        # autograd.grad() call has no graph left to backward through and
        # raises "Trying to backward through the graph a second time".
        def fvp(v):
            kl = self.policy.kl(obs, old_mu, old_std)  # fresh forward pass every call

            kl_grads = torch.autograd.grad(
                kl,
                params,
                create_graph=True,   # required: builds a differentiable graph
                retain_graph=True,
            )
            flat_kl_grad = torch.cat([kg.reshape(-1) for kg in kl_grads])

            grad_v = torch.dot(flat_kl_grad, v)

            hvps = torch.autograd.grad(
                grad_v,
                params,
                retain_graph=False,  # this graph is single-use, safe to free
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
            self.policy.set_flat_params(old_params)  # reject the update

        # ---- value function regression (several epochs of simple MSE fit) ---- #
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