import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt  # CHANGED: for plotting reward curve

from ppo import GaussianPolicy, ValueNet, PPOBuffer, RunningNorm  # CHANGED: added RunningNorm

# ----------------------------------------------------------------------
# Import YOUR custom env from env.py.
# Change "PusherEnv" below if your class inside env.py has a different name.
# ----------------------------------------------------------------------
from env import PusherEnv


def make_env():
    return PusherEnv()


def ppo_train(
    epochs=500,
    steps_per_epoch=4000,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    target_kl=0.01,
    hidden=(128, 128),
    max_ep_len=100,
    ent_coef=0.0,  
    save_path="ppo_pusher_policy.pt",
    log_every=1,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    pi_net = GaussianPolicy(obs_dim, act_dim, hidden)
    v_net = ValueNet(obs_dim, hidden)
    pi_optimizer = optim.Adam(pi_net.parameters(), lr=pi_lr)
    v_optimizer = optim.Adam(v_net.parameters(), lr=vf_lr)

    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    obs_rms = RunningNorm(obs_dim)  # CHANGED: added, normalizes observations

    def compute_loss_pi(data, old_logp):
        obs, act, adv = data['obs'], data['act'], data['adv']
        dist, logp = pi_net(obs, act)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        ent = dist.entropy().sum(axis=-1).mean()
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - ent_coef * ent
        approx_kl = (old_logp - logp).mean().item()
        return loss_pi, approx_kl, ent.item()

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((v_net(obs) - ret) ** 2).mean()

    def update():
        data = buf.get()
        old_logp = data['logp']

        kl, ent = 0.0, 0.0
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, kl, ent = compute_loss_pi(data, old_logp)
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        for i in range(train_v_iters):
            v_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            v_optimizer.step()

        return loss_pi.item(), loss_v.item(), kl, ent

    obs, _ = env.reset(seed=seed)
    obs_rms.update(obs)         # CHANGED
    obs = obs_rms.normalize(obs)  # CHANGED
    ep_ret, ep_len = 0.0, 0
    best_avg_ret = -np.inf
    start_time = time.time()
    reward_history = []  # CHANGED: track avg return per epoch, for plotting later

    for epoch in range(epochs):
        ep_returns, ep_lens = [], []

        for t in range(steps_per_epoch):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = pi_net.distribution(obs_t)
                act = dist.sample()
                logp = pi_net.log_prob(dist, act)
                val = v_net(obs_t)

            act_np = act.numpy()
            act_clipped = np.clip(act_np, act_low, act_high)

            next_obs, rew, terminated, truncated, _ = env.step(act_clipped)
            done = terminated or truncated
            ep_ret += rew
            ep_len += 1

            buf.store(obs, act_np, rew, val.item(), logp.item())

            obs_rms.update(next_obs)          # CHANGED (was: obs = next_obs)
            obs = obs_rms.normalize(next_obs)  # CHANGED

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or epoch_ended:
                if timeout or epoch_ended or not terminated:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        last_val = v_net(obs_t).item()
                else:
                    last_val = 0.0
                buf.finish_path(last_val)

                if terminal:
                    ep_returns.append(ep_ret)
                    ep_lens.append(ep_len)

                obs, _ = env.reset()
                obs_rms.update(obs)         
                obs = obs_rms.normalize(obs)  
                ep_ret, ep_len = 0.0, 0

        loss_pi, loss_v, kl, ent = update()

        avg_ret = np.mean(ep_returns) if ep_returns else float('nan')
        avg_len = np.mean(ep_lens) if ep_lens else float('nan')
        elapsed = time.time() - start_time
        reward_history.append(avg_ret)  # CHANGED

        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | AvgRet {avg_ret:8.2f} | AvgLen {avg_len:6.1f} "
                  f"| LossPi {loss_pi:.4f} | LossV {loss_v:.4f} | KL {kl:.4f} "
                  f"| Ent {ent:.4f} | Time {elapsed:6.1f}s")

        if not np.isnan(avg_ret) and avg_ret > best_avg_ret:
            best_avg_ret = avg_ret
            torch.save(pi_net.state_dict(), save_path.replace(".pt", "_best.pt"))

    torch.save(pi_net.state_dict(), save_path)
    torch.save(v_net.state_dict(), save_path.replace("policy", "value"))
    obs_rms.save(save_path.replace(".pt", "_obs_rms.pt"))  # CHANGED: save normalizer for inference (.pt now)
    torch.save(reward_history,save_path.replace(".pt","_reward_history.pt"))
    print(f"\nTraining complete. Best avg return: {best_avg_ret:.2f}")
    print(f"Saved: {save_path}")

    # CHANGED: plot reward curve and save as png
    clean_history = [r for r in reward_history if not np.isnan(r)]
    plt.figure(figsize=(8, 5))
    plt.plot(clean_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plot_path = save_path.replace(".pt", "_reward_curve.png")
    plt.savefig(plot_path)
    print(f"Saved reward plot: {plot_path}")

    env.close()
    return pi_net, v_net


if __name__ == "__main__":
    ppo_train(
        epochs=500,
        steps_per_epoch=4000,
        max_ep_len=100,
    )