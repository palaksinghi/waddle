"""Run a trained custom HalfCheetah PPO policy, optionally in the MuJoCo viewer."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from half_cheetah_env import HalfCheetahEnv
from ppo import ActorCritic, ObservationNormalizer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    env = HalfCheetahEnv(render_mode="human" if args.render else None)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = ActorCritic(obs_dim, action_dim).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()

    # Inference must use the training observation statistics, without updating them.
    obs_normalizer = ObservationNormalizer((obs_dim,), args.obs_clip)
    obs_normalizer.rms.mean = np.asarray(checkpoint["obs_mean"], dtype=np.float64)
    obs_normalizer.rms.var = np.asarray(checkpoint["obs_var"], dtype=np.float64)

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        components = {"reward_forward": 0.0, "reward_healthy": 0.0, "reward_ctrl_penalty": 0.0}
        for _ in range(args.max_steps):
            normalized_obs = obs_normalizer.normalize(obs)
            with torch.no_grad():
                # PPO's actor output is the Gaussian mean: deterministic evaluation action.
                action = agent.actor(torch.as_tensor(normalized_obs, device=device).unsqueeze(0))[0].cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            for key in components:
                components[key] += info[key]
            if terminated or truncated:
                break
        returns.append(total_reward)
        print(f"episode {episode + 1}: return={total_reward:.2f}, steps={_ + 1}, "
              f"forward={components['reward_forward']:.2f}, healthy={components['reward_healthy']:.2f}, "
              f"control_penalty={components['reward_ctrl_penalty']:.2f}")
    env.close()
    print(f"mean return over {args.episodes} episode(s): {np.mean(returns):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference/evaluation for the custom HalfCheetah PPO policy")
    parser.add_argument("--checkpoint", default="checkpoints/halfcheetah_ppo.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--obs-clip", type=float, default=10.0)
    parser.add_argument("--render", action="store_true", help="Open the live MuJoCo viewer")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
