"""
Run a trained TRPO policy in the MuJoCo humanoid environment and watch it
walk, either in an interactive viewer window or headless (saving a video).

Usage:
    python inference.py --checkpoint checkpoints/policy_final.pt
    python inference.py --checkpoint checkpoints/policy_final.pt --video out.mp4 --no-render
"""

import argparse

import numpy as np
import torch

from humanoid_env import make_env
from trpo import GaussianPolicy, device


def run_episode(env, policy, deterministic=True, max_steps=1000, record_frames=False):
    obs, _ = env.reset()
    total_reward = 0.0
    frames = []

    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act = policy.act(obs_t, deterministic=deterministic).squeeze(0).cpu().numpy()
        act = np.clip(act, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, _ = env.step(act)
        total_reward += reward

        if record_frames:
            frames.append(env.render())

        if terminated or truncated:
            break

    return total_reward, step + 1, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    parser.add_argument("--no-render", action="store_true", help="disable the interactive viewer window")
    parser.add_argument("--video", type=str, default=None, help="path to save an mp4 (requires --no-render)")
    args = parser.parse_args()

    render_mode = "rgb_array" if (args.no_render and args.video) else ("human" if not args.no_render else None)
    env = make_env(render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    all_frames = []
    for ep in range(args.episodes):
        ret, length, frames = run_episode(
            env, policy,
            deterministic=not args.stochastic,
            max_steps=args.max_steps,
            record_frames=bool(args.video),
        )
        print(f"episode {ep + 1}: return={ret:.2f}  length={length}")
        all_frames.extend(frames)

    if args.video and all_frames:
        import imageio
        imageio.mimsave(args.video, all_frames, fps=env.metadata.get("render_fps", 30))
        print(f"Saved video to {args.video}")

    env.close()


if __name__ == "__main__":
    main()