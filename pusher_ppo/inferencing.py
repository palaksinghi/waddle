import time
import numpy as np
import torch

from ppo import GaussianPolicy, RunningNorm
from env import PusherEnv  # change name if your class in env.py is different


def run_inference(
    model_path="ppo_pusher_policy_best.pt",
    obs_rms_path="ppo_pusher_policy_obs_rms.pt",  # NEW: saved normalizer stats
    episodes=5,
    max_ep_len=100,
    render=True,
    deterministic=True,
):
    env = PusherEnv(render_mode="human" if render else None)  # FIX: pass render_mode

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    pi_net = GaussianPolicy(obs_dim, act_dim)
    pi_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    pi_net.eval()

    # NEW: load the same obs normalizer stats used during training.
    # Without this, the policy sees obs on a totally different scale
    # than it was trained on, and will act randomly/incorrectly.
    obs_rms = RunningNorm(obs_dim)
    obs_rms.load(obs_rms_path)

    print(f"Loaded model from {model_path}")
    print(f"Loaded obs normalizer from {obs_rms_path}")
    print(f"Running {episodes} episodes | deterministic={deterministic}")

    all_returns = []

    for ep in range(episodes):
        raw_obs, _ = env.reset()
        obs = obs_rms.normalize(raw_obs)  # NEW
        ep_ret, ep_len = 0.0, 0
        done = False

        while not done and ep_len < max_ep_len:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                dist = pi_net.distribution(obs_t)
                if deterministic:
                    act = dist.mean          # use mean action (no exploration noise)
                else:
                    act = dist.sample()      # stochastic action

            act_np = np.clip(act.numpy(), act_low, act_high)
            raw_obs, rew, terminated, truncated, _ = env.step(act_np)
            obs = obs_rms.normalize(raw_obs)  # NEW: normalize before next policy call
            done = terminated or truncated

            ep_ret += rew
            ep_len += 1

            if render:
                env.render()
                time.sleep(0.02)  # slow down for viewing; remove/reduce if too slow

        all_returns.append(ep_ret)
        print(f"Episode {ep + 1:3d} | Return {ep_ret:8.2f} | Length {ep_len}")

    env.close()

    print(f"\nAverage return over {episodes} episodes: {np.mean(all_returns):.2f} "
          f"(+/- {np.std(all_returns):.2f})")


if __name__ == "__main__":
    run_inference(
        model_path="ppo_pusher_policy_best.pt",
        obs_rms_path="ppo_pusher_policy_obs_rms.pt",
        episodes=5,
        render=True,
        deterministic=True,
    )