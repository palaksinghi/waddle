import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
# Policy network (must match the architecture used in trpo_train.py)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """ outputs a diagonal Gaussian over continuous actions."""

    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def sample(self, obs):
        dist = self.forward(obs)
        act = dist.sample()
        logp = dist.log_prob(act).sum(axis=-1)
        return act, logp
# Inference

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    #render_mode = "human" if args.render else None
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="inference_videos",
        episode_trigger=lambda episode_id: True,
        name_prefix="humanoid"
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    #policy.load_state_dict(torch.load(args.checkpoint, map_location=device),weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()

    all_rewards = []
    for ep in range(args.episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if args.deterministic:
                    action = policy(state_t).mean.squeeze(0).cpu().numpy()
                else:
                    action, _ = policy.sample(state_t)
                    action = action.squeeze(0).cpu().numpy()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
    
        all_rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{args.episodes}: reward = {ep_reward:.2f}  steps = {steps}")

    print(f"\nMean reward over {args.episodes} episodes: {np.mean(all_rewards):.2f}")
    print(f"Std  reward over {args.episodes} episodes: {np.std(all_rewards):.2f}")
    env.close()

def build_argparser():
    p = argparse.ArgumentParser(description="TRPO inference on Humanoid-v5 (single environment)")
    p.add_argument("--checkpoint", type=str, required=True, help="path to saved policy .pt file")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--render", action="store_true", help="render the environment (requires a display)")
    p.add_argument("--deterministic", action="store_true", help="use mean action instead of sampling")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    infer(args)