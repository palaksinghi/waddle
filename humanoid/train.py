"""
Train TRPO on the custom MuJoCo humanoid.

Usage:
    python train.py --iters 500 --steps-per-iter 4000

Checkpoints are written to ./checkpoints/policy_<iter>.pt and the
final policy is also saved to ./checkpoints/policy_final.pt
"""

import argparse
import os
import time

import numpy as np
import torch

from humanoid_env import make_env
from trpo import TRPO, compute_gae, device


def collect_rollout(env, agent, steps_per_iter):
    """Collect a batch of ~steps_per_iter environment steps (possibly across several episodes)."""
    obs_buf, act_buf, rew_buf, done_buf, val_buf = [], [], [], [], []
    logp_buf, mu_buf, std_buf = [], [], []
    ep_returns, ep_lens = [], []

    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0

    for _ in range(steps_per_iter):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = agent.policy(obs_t)
            act_t = dist.sample()
            logp_t = dist.log_prob(act_t).sum(-1)
            val_t = agent.value(obs_t)

        act = act_t.squeeze(0).cpu().numpy()
        clipped_act = np.clip(act, env.action_space.low, env.action_space.high)

        next_obs, rew, terminated, truncated, _ = env.step(clipped_act)
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(act)
        rew_buf.append(rew)
        done_buf.append(float(done))
        val_buf.append(val_t.item())
        logp_buf.append(logp_t.item())
        mu_buf.append(dist.mean.squeeze(0).cpu().numpy())
        std_buf.append(dist.stddev.squeeze(0).cpu().numpy())

        ep_ret += rew
        ep_len += 1
        obs = next_obs

        if done:
            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

    # bootstrap value for the (possibly unfinished) trailing trajectory
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_val = agent.value(obs_t).item() if not done_buf[-1] else 0.0

    adv, ret = compute_gae(
        np.array(rew_buf, dtype=np.float32),
        np.array(val_buf, dtype=np.float32),
        np.array(done_buf, dtype=np.float32),
        last_val,
        gamma=agent.gamma,
        lam=agent.lam,
    )

    batch = dict(
        obs=torch.as_tensor(np.array(obs_buf), dtype=torch.float32),
        act=torch.as_tensor(np.array(act_buf), dtype=torch.float32),
        adv=torch.as_tensor(adv, dtype=torch.float32),
        ret=torch.as_tensor(ret, dtype=torch.float32),
        logp=torch.as_tensor(np.array(logp_buf), dtype=torch.float32),
        mu=torch.as_tensor(np.array(mu_buf), dtype=torch.float32),
        std=torch.as_tensor(np.array(std_buf), dtype=torch.float32),
    )
    stats = dict(
        mean_return=float(np.mean(ep_returns)) if ep_returns else float("nan"),
        n_episodes=len(ep_returns),
        mean_len=float(np.mean(ep_lens)) if ep_lens else steps_per_iter,
    )
    return batch, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--steps-per-iter", type=int, default=4000)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TRPO(
        obs_dim, act_dim,
        max_kl=args.max_kl, damping=args.damping,
        gamma=args.gamma, lam=args.lam,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        batch, stats = collect_rollout(env, agent, args.steps_per_iter)
        info = agent.update(
            batch["obs"], batch["act"], batch["adv"], batch["ret"],
            batch["logp"], batch["mu"], batch["std"],
        )

        elapsed = time.time() - t0
        print(
            f"iter {it:4d} | return {stats['mean_return']:8.2f} | "
            f"episodes {stats['n_episodes']:3d} | KL {info['kl']:.4f} | "
            f"vloss {info['value_loss']:8.3f} | accepted {info['step_accepted']} | "
            f"time {elapsed:6.1f}s"
        )

        if it % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"policy_{it}.pt")
            torch.save(agent.policy.state_dict(), ckpt_path)

    final_path = os.path.join(args.out_dir, "policy_final.pt")
    torch.save(agent.policy.state_dict(), final_path)
    print(f"Training complete. Final policy saved to {final_path}")
    env.close()


if __name__ == "__main__":
    main()