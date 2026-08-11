import os
import math
import time
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

# Importing custom modules
from build_env import c_env  
from ppo import Agent, RolloutBuffer


def make_env_thunk(seed, render_mode=None):
    def thunk():
        env = c_env(render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
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
    total_timesteps = 1_000_000    
    num_envs = 4                # Number of parallel environments
    num_steps = 2048             # Rollout steps per environment
    update_epochs = 10           # PPO update epochs per rollout batch
    num_minibatches = 32         # Mini-batches for SGD    
    gamma = 0.99                 # Discount factor
    gae_lambda = 0.95            # GAE lambda
    clip_coef = 0.2              # PPO clipping parameter epsilon
    vf_coef = 0.5                # Value loss coefficient
    ent_coef = 0.0               # Entropy bonus coefficient
    max_grad_norm = 0.5          # Gradient norm clipping threshold
    target_kl = 0.015            # Target KL divergence threshold

    out_dir = "./checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_updates = total_timesteps // batch_size

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
    
    # FIX: Reset returns (obs, info) tuple
    next_obs_np, _ = env.reset(seed=seed)
    next_obs = torch.Tensor(next_obs_np).to(device=device)
    
    # FIX: Initialize clean zero tensor
    next_done = torch.zeros(num_envs, device=device)

    start_time = time.time()

    """
    TRAINING
    """
    for update in range(1, num_updates + 1):

        # --- 1. ROLLOUT COLLECTION ---
        for step in range(num_steps): 
            buffer.obs[step] = next_obs
            buffer.dones[step] = next_done
            
            with torch.no_grad():
                # FIX: Used agent instance and correct method name
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                buffer.values[step] = value.flatten()

            buffer.actions[step] = action 
            buffer.logprobs[step] = logprob

            # Step environment
            next_obs_np, reward, terminations, truncations, infos = env.step(action.cpu().numpy())
            dones = np.logical_or(terminations, truncations)

            next_obs = torch.Tensor(next_obs_np).to(device=device)
            next_done = torch.Tensor(dones, device=device)
            buffer.rewards[step] = torch.tensor(reward, device=device).view(-1)

        # --- 2. GAE ADVANTAGE CALCULATION ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            buffer.compute_gae(next_value, next_done, gamma, gae_lambda)

        b_obs = buffer.obs.reshape(-1, obs_dim)
        b_actions = buffer.actions.reshape(-1, action_dim)
        b_advantages = buffer.advantages.reshape(-1)
        b_logprobs = buffer.logprobs.reshape(-1)
        b_values = buffer.values.reshape(-1)
        
        # FIX: Correct buffer property reference
        b_returns = buffer.returns.reshape(-1)

        b_inds = np.arange(batch_size)
        
        # FIX: Defined kl_break flag before epoch loop
        kl_break = False

        # --- 3. PPO OPTIMIZATION ---
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # FIX: Called agent instance
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = newvalue.view(-1)

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                logratio = newlogprob - b_logprobs[mb_inds]
                
                # FIX: Used logratio.exp()
                ratio = logratio.exp()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Policy loss
                policy_loss = -mb_advantages * ratio
                policy_loss1 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                
                # FIX: Added .mean() scalar reduction
                pg_loss = torch.max(policy_loss, policy_loss1).mean()

                if target_kl is not None and approx_kl > target_kl:
                    kl_break = True
                    break

                # Entropy loss
                entropy_loss = entropy.mean()

                # Value loss
                value_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Loss
                loss = pg_loss + (vf_coef * value_loss) - (ent_coef * entropy_loss)
                
                optimizer.zero_grad()
                loss.backward()
                
                # FIX: Called agent.parameters()
                nn.utils.clip_grad_norm_(agent.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            if kl_break:
                break

        # FIX: Correct indentation for per-update metrics logging
        sps = int((update * batch_size) / (time.time() - start_time))
        print(
            f"Iter [{update:3d}/{num_updates}] | "
            f"SPS: {sps:5d} | "
            f"KL: {approx_kl.item():.4f} | "
            f"PG Loss: {pg_loss.item():.4f} | "
            f"V Loss: {value_loss.item():.4f}"
        )

    # FIX: Correct indentation for saving and closing (after training finishes)
    final_path = os.path.join(out_dir, "walker2d_ppo_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"✅ Training complete! Model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    train()