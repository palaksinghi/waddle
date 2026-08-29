import os
import time
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

# Importing custom modules
from build_env import CustomWalker_2dEnv
from ppo import Agent, RolloutBuffer


def make_env_thunk(seed, render_mode=None):
    def thunk():
        env = CustomWalker_2dEnv(render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def train():
    """
    HYPERPARAMETERS
    """
    seed = 41
    learning_rate = 3e-4
    num_envs = 1                     # Number of parallel environments
    num_steps = 2048                 # Rollout steps per environment
    update_epochs = 10               # PPO update epochs per rollout batch
    num_minibatches = 32             # Mini-batches for SGD (clean divisor of batch_size)
    gamma = 0.99                     # Discount factor
    gae_lambda = 0.95                # GAE lambda
    clip_coef = 0.2                  # PPO clipping parameter epsilon
    vf_coef = 0.5                    # Value loss coefficient
    ent_coef = 0.01                  # Entropy bonus - avoids premature convergence
                                      # to a "give up and fall" local optimum
    max_grad_norm = 0.5               # Gradient norm clipping threshold
    target_kl = 0.02                  # Slightly relaxed from 0.015 so PPO epochs
                                       # aren't cut short almost every update

    checkpoint_every = 10              # Save an intermediate checkpoint every N updates

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    # Walker2d is a hard continuous-control locomotion task - it needs many
    # millions of timesteps to learn real walking, not just a few. 3000
    # updates * 8192 batch_size = ~24.5M timesteps, a realistic budget.
    num_updates = 500
    total_timesteps = num_updates * batch_size

    out_dir = "./checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    VECTORIZED SYNC ENVIRONMENT
    """
    env = gym.vector.SyncVectorEnv([make_env_thunk(seed + i) for i in range(num_envs)])

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    agent = Agent(env).to(device=device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    buffer = RolloutBuffer(num_envs=num_envs, num_steps=num_steps, obs_dim=obs_dim, device=device, action_dim=action_dim)

    next_obs_np, _ = env.reset(seed=seed)
    next_obs = torch.Tensor(next_obs_np).to(device=device)
    next_done = torch.zeros(num_envs, device=device)

    start_time = time.time()

    # Track recent episode returns/lengths to monitor real training progress
    recent_returns = []
    recent_lengths = []

    """
    TRAINING
    """
    for update in range(1, num_updates + 1):

        # Linear learning-rate annealing: helps stabilize late-stage training
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * learning_rate

        # --- 1. ROLLOUT COLLECTION ---
        for step in range(num_steps):
            buffer.obs[step] = next_obs
            buffer.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                buffer.values[step] = value.flatten()

            buffer.actions[step] = action
            buffer.logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = env.step(action.cpu().numpy())
            dones = np.logical_or(terminations, truncations)

            next_obs = torch.Tensor(next_obs_np).to(device=device)
            next_done = torch.tensor(dones, device=device, dtype=torch.float32)
            buffer.rewards[step] = torch.tensor(reward, device=device).view(-1)

            # RecordEpisodeStatistics puts finished-episode stats in
            # infos["episode"], masked by which sub-envs just finished.
            if "episode" in infos:
                ep_mask = infos["episode"]["_r"] if "_r" in infos["episode"] else np.array(dones)
                finished_returns = infos["episode"]["r"][ep_mask]
                finished_lengths = infos["episode"]["l"][ep_mask]
                recent_returns.extend(finished_returns.tolist())
                recent_lengths.extend(finished_lengths.tolist())

        # --- 2. GAE ADVANTAGE CALCULATION ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            buffer.compute_gae(next_value, next_done, gamma, gae_lambda)

        b_obs = buffer.obs.reshape(-1, obs_dim)
        b_actions = buffer.actions.reshape(-1, action_dim)
        b_advantages = buffer.advantages.reshape(-1)
        b_logprobs = buffer.logprobs.reshape(-1)
        b_returns = buffer.returns.reshape(-1)

        b_inds = np.arange(batch_size)
        kl_break = False
        approx_kl = torch.tensor(0.0)
        pg_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)

        # --- 3. PPO OPTIMIZATION ---
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = newvalue.view(-1)

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                policy_loss = -mb_advantages * ratio
                policy_loss1 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(policy_loss, policy_loss1).mean()

                if target_kl is not None and approx_kl > target_kl:
                    kl_break = True
                    break

                entropy_loss = entropy.mean()
                value_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss + (vf_coef * value_loss) - (ent_coef * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            if kl_break:
                break

        sps = int((update * batch_size) / (time.time() - start_time))
        mean_return = np.mean(recent_returns[-20:]) if recent_returns else float("nan")
        mean_length = np.mean(recent_lengths[-20:]) if recent_lengths else float("nan")
        print(
            f"Iter [{update:4d}/{num_updates}] | "
            f"SPS: {sps:5d} | "
            f"EpReturn: {mean_return:8.2f} | "
            f"EpLen: {mean_length:6.1f} | "
            f"KL: {approx_kl.item():.4f} | "
            f"PG Loss: {pg_loss.item():.4f} | "
            f"V Loss: {value_loss.item():.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Periodic checkpointing so you can evaluate progress and roll back
        # to an earlier, better checkpoint if performance degrades later on.
        if update % checkpoint_every == 0:
            ckpt_path = os.path.join(out_dir, f"walker2d_ppo_update{update}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    final_path = os.path.join(out_dir, "walker2d_ppo_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"✅ Training complete! Model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    train()