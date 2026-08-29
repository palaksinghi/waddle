"""Train custom HalfCheetah with PPO. Add --render to watch deterministic rollouts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from half_cheetah_env import HalfCheetahEnv
from ppo import ActorCritic, ObservationNormalizer, RewardNormalizer, ValueNormalizer


def make_env(seed):
    def thunk():
        env = HalfCheetahEnv(); env = gym.wrappers.TimeLimit(env, max_episode_steps=1000); env.reset(seed=seed); return env
    return thunk


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(args.seed + i) for i in range(args.num_envs)])
    obs, _ = envs.reset(seed=args.seed)
    obs_dim, act_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    agent = ActorCritic(obs_dim, act_dim).to(device); optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_norm = ObservationNormalizer((obs_dim,), args.obs_clip)
    reward_norm = RewardNormalizer(args.num_envs, args.gamma, args.reward_clip)
    value_norm = ValueNormalizer()
    steps_per_update = args.num_steps * args.num_envs; updates = args.total_timesteps // steps_per_update
    run_dir = Path(args.output_dir); run_dir.mkdir(parents=True, exist_ok=True)
    episode_returns = np.zeros(args.num_envs)

    for update in range(1, updates + 1):
        obs_buf = np.zeros((args.num_steps, args.num_envs, obs_dim), np.float32); act_buf = np.zeros((args.num_steps, args.num_envs, act_dim), np.float32)
        logp_buf = np.zeros((args.num_steps, args.num_envs), np.float32); rew_buf = np.zeros((args.num_steps, args.num_envs), np.float32)
        done_buf = np.zeros((args.num_steps, args.num_envs), np.float32); val_buf = np.zeros((args.num_steps, args.num_envs), np.float32)
        for step in range(args.num_steps):
            obs_norm.update(obs); norm_obs = obs_norm.normalize(obs); obs_buf[step] = norm_obs
            with torch.no_grad():
                action, logp, _, normalized_value = agent.get_action_and_value(torch.as_tensor(norm_obs, device=device))
                action = action.cpu().numpy(); logp_buf[step] = logp.cpu().numpy()
                val_buf[step] = value_norm.denormalize(normalized_value.cpu().numpy())
            action = np.clip(action, envs.single_action_space.low, envs.single_action_space.high); act_buf[step] = action
            obs, reward, terminated, truncated, _ = envs.step(action); done = np.logical_or(terminated, truncated)
            rew_buf[step] = reward_norm(reward, done); done_buf[step] = done
            episode_returns += reward
            if done.any():
                for value in episode_returns[done]: print(f"episode return: {value:8.2f}")
                episode_returns[done] = 0.0

        with torch.no_grad():
            next_value = value_norm.denormalize(agent.get_action_and_value(torch.as_tensor(obs_norm.normalize(obs), device=device))[3].cpu().numpy())
        advantages = np.zeros_like(rew_buf); last_gae = np.zeros(args.num_envs)
        for t in reversed(range(args.num_steps)):
            nonterminal = 1.0 - done_buf[t]; next_val = next_value if t == args.num_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_val * nonterminal - val_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * nonterminal * last_gae; advantages[t] = last_gae
        returns = advantages + val_buf
        value_norm.update(returns)

        b_obs, b_act = obs_buf.reshape(-1, obs_dim), act_buf.reshape(-1, act_dim)
        b_logp, b_adv, b_ret, b_val = logp_buf.ravel(), advantages.ravel(), returns.ravel(), val_buf.ravel()
        indices = np.arange(steps_per_update)
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, steps_per_update, args.minibatch_size):
                mb = indices[start:start + args.minibatch_size]
                mb_obs, mb_act = torch.as_tensor(b_obs[mb], device=device), torch.as_tensor(b_act[mb], device=device)
                _, new_logp, entropy, new_value_norm = agent.get_action_and_value(mb_obs, mb_act)
                ratio = (new_logp - torch.as_tensor(b_logp[mb], device=device)).exp()
                adv = torch.as_tensor(b_adv[mb], device=device); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * adv).mean()
                old_value_norm = torch.as_tensor(value_norm.normalize(b_val[mb]), dtype=torch.float32, device=device)
                target_value_norm = torch.as_tensor(value_norm.normalize(b_ret[mb]), dtype=torch.float32, device=device)
                clipped_value = old_value_norm + torch.clamp(new_value_norm - old_value_norm, -args.value_clip_coef, args.value_clip_coef)
                value_loss = 0.5 * torch.maximum((new_value_norm - target_value_norm).square(), (clipped_value - target_value_norm).square()).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # gradient clipping
                optimizer.step()
        if update % args.save_every == 0 or update == updates:
            torch.save({"model": agent.state_dict(), "obs_mean": obs_norm.rms.mean, "obs_var": obs_norm.rms.var, "value_mean": value_norm.rms.mean, "value_var": value_norm.rms.var}, run_dir / "halfcheetah_ppo.pt")
        print(f"update {update}/{updates} | mean rollout reward: {rew_buf.sum(0).mean():.2f}")
    envs.close()
    if args.render: render(args, agent, obs_norm, device)


def render(args, agent, obs_norm, device):
    env = HalfCheetahEnv(render_mode="human"); obs, _ = env.reset(seed=args.seed); total = 0.0
    for _ in range(args.render_steps):
        with torch.no_grad(): action = agent.get_action_and_value(torch.as_tensor(obs_norm.normalize(obs), device=device).unsqueeze(0))[0][0].cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(np.clip(action, -1, 1)); total += reward
        if terminated or truncated: obs, _ = env.reset()
    env.close(); print(f"render return: {total:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000); parser.add_argument("--num-envs", type=int, default=8); parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--update-epochs", type=int, default=10); parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99); parser.add_argument("--gae-lambda", type=float, default=0.95); parser.add_argument("--clip-coef", type=float, default=0.2); parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5); parser.add_argument("--ent-coef", type=float, default=0.0); parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-clip", type=float, default=10.0); parser.add_argument("--obs-clip", type=float, default=10.0); parser.add_argument("--seed", type=int, default=1); parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints"); parser.add_argument("--save-every", type=int, default=10); parser.add_argument("--render", action="store_true"); parser.add_argument("--render-steps", type=int, default=2000)
    main(parser.parse_args())
