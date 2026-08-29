"""
Inference script for your custom PPO-trained Walker2d agent
(CustomWalker_2dEnv + Agent from ppo.py).

Usage:
    python inference.py --model checkpoints/walker2d_ppo_final.pt
    python inference.py --model checkpoints/walker2d_ppo_final.pt --episodes 10 --no-render
    python inference.py --model checkpoints/walker2d_ppo_final.pt --stochastic
"""

import argparse
import numpy as np
import torch

from build_env import CustomWalker_2dEnv
from ppo import Agent


def build_env(render):
    render_mode = "human" if render else None
    return CustomWalker_2dEnv(render_mode=render_mode)


def run_inference(model_path, episodes, render, deterministic, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = build_env(render)

    agent = Agent(env).to(device)
    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    ep_rewards = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                if deterministic:
                    # Use the mean action directly (best for evaluation, no sampling noise)
                    action = agent.actor_mean(obs_t)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        ep_rewards.append(total_reward)
        print(f"Episode {ep + 1}/{episodes}: reward = {total_reward:.2f}, steps = {steps}")

    env.close()
    print(f"\nMean reward over {episodes} episodes: {np.mean(ep_rewards):.2f} "
          f"+/- {np.std(ep_rewards):.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained custom Walker2d PPO agent")
    p.add_argument("--model", required=True, help="Path to the saved checkpoint (.pt state_dict)")
    p.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    p.add_argument("--render", dest="render", action="store_true", default=True,
                   help="Render the environment (default: on)")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Disable rendering (faster evaluation)")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy distribution instead of using the mean")
    p.add_argument("--seed", type=int, default=41, help="Base seed for env resets")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model,
        episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        seed=args.seed,
    )