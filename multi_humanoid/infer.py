"""
Run inference with a trained TRPO policy checkpoint (.pth) on the shared
MultiHumanoidEnv sim, with a live MuJoCo viewer window by default.

This does NOT train anything -- no TRPO update, no value net needed. It
just loads GaussianPolicy weights and rolls the policy out, either
deterministically (dist.mean, no sampling noise -- best for watching
"final" behavior) or stochastically (dist.sample() -- matches training
distribution, useful for checking exploration/variance).

Usage:
    # Watch the final trained policy live
    python3 infer.py --checkpoint checkpoints/policy_final.pt --num-agents 8

    # Watch a specific checkpoint, run for a fixed number of steps, no window
    # (headless -- useful for numeric eval only)
    python3 infer.py --checkpoint checkpoints/policy_250.pt --num-agents 8 \
        --no-render --episodes 5

    # Sample actions instead of using the mean action (stochastic policy)
    python3 infer.py --checkpoint checkpoints/policy_final.pt --stochastic
"""

import argparse
import time

import numpy as np
import torch

from multi_humanoid_env import MultiHumanoidEnv
from trpo import GaussianPolicy, device


def run_inference(policy, env, steps, deterministic=True, render=True, real_time=True):
    """
    Roll the policy out in env for `steps` timesteps (a single shared-sim
    rollout covering all agents at once, same as training). Prints episode
    returns as agents finish (fall + auto-reset in place, per MultiHumanoidEnv).
    """
    n_agents = env.n_agents
    obs, _ = env.reset()

    running_ret = np.zeros(n_agents, dtype=np.float32)
    running_len = np.zeros(n_agents, dtype=np.int64)
    ep_returns, ep_lens = [], []

    policy.eval()

    for t in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = policy(obs_t)
            act_t = dist.mean if deterministic else dist.sample()

        act = np.clip(act_t.numpy(), -1.0, 1.0)
        obs, rew, terminated, truncated, _ = env.step(act)
        done = np.logical_or(terminated, truncated)

        running_ret += rew
        running_len += 1
        for i in range(n_agents):
            if done[i]:
                ep_returns.append(running_ret[i])
                ep_lens.append(running_len[i])
                print(f"  agent {i}: episode return {running_ret[i]:8.2f}  length {running_len[i]}")
                running_ret[i] = 0.0
                running_len[i] = 0

        if render and real_time:
            # MultiHumanoidEnv.step() already calls self.render() internally
            # when render_mode == "human", but we sleep here to roughly match
            # sim dt so the viewer doesn't run faster than real time.
            time.sleep(env.dt)

    return ep_returns, ep_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="path to a .pt/.pth policy state_dict saved by train_multi_agent.py")
    parser.add_argument("--num-agents", type=int, default=8,
                         help="must match the --num-agents used during training "
                              "(the checkpoint's obs/act dims depend on it)")
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=2000,
                         help="how many timesteps to roll out")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample actions instead of using the deterministic mean action")
    parser.add_argument("--no-render", action="store_true",
                         help="run headless (no live viewer window), just prints episode returns")
    parser.add_argument("--no-real-time", action="store_true",
                         help="don't sleep to match wall-clock/sim time; run render as fast as possible")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    render = not args.no_render
    env = MultiHumanoidEnv(
        n_agents=args.num_agents,
        spacing=args.spacing,
        render_mode="human" if render else None,
    )

    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim).to("cpu")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  num_agents={args.num_agents}  "
          f"mode={'stochastic' if args.stochastic else 'deterministic'}")

    ep_returns, ep_lens = run_inference(
        policy, env,
        steps=args.steps,
        deterministic=not args.stochastic,
        render=render,
        real_time=render and not args.no_real_time,
    )

    if ep_returns:
        print(f"\n{len(ep_returns)} episodes finished | "
              f"mean return {np.mean(ep_returns):8.2f} | "
              f"mean length {np.mean(ep_lens):6.1f}")
    else:
        print("\nNo episodes finished within --steps (agents stayed healthy the whole rollout).")

    env.close()


if __name__ == "__main__":
    main()