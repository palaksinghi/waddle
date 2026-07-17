"""
Train TRPO with N humanoids sharing ONE MuJoCo simulation (MultiHumanoidEnv),
instead of N separate single-robot subprocess envs.

Differences from the AsyncVectorEnv version:
  - No subprocess pool: one process, one mj_step() call per frame advances
    every agent's physics together. This trades multi-core parallelism for
    lower per-step Python/IPC overhead, and lets robots physically interact
    if placed close enough (set `spacing` in MultiHumanoidEnv accordingly).
  - Because all agents live in one sim, "vectorizing" the rollout loop is
    just calling env.step() once per timestep with a (n_agents, act_dim)
    action batch -- there's no vector env wrapper needed at all.
  - If you want BOTH multi-core parallelism and multiple robots per sim,
    wrap MultiHumanoidEnv itself in AsyncVectorEnv (K processes, each
    running M agents = K*M total agents). See note at bottom of file.

Live rendering:
  - Pass --render to open a live MuJoCo viewer window during training.
    This calls env.step() with render_mode="human", which syncs the
    viewer once per env.step() (i.e. once per `frame_skip` substeps),
    not once per substep -- syncing every substep would be far too slow.
  - Live rendering runs at roughly real-time / wall-clock speed and will
    noticeably slow down training. Use --render-every N to only render
    one iteration out of every N (viewer opens for that iteration's
    rollout only), or omit --render entirely for full-speed headless
    training.
  - Live rendering requires a real display (a local monitor, or X11
    forwarding if running remotely). It will not work on a fully
    headless server with no display attached.

Usage:
    python train_multi_agent.py --iters 500 --steps-per-iter 4000 --num-agents 8
    python train_multi_agent.py --iters 500 --num-agents 8 --render
    python train_multi_agent.py --iters 500 --num-agents 8 --render --render-every 10
"""

import argparse
import os
import time

import numpy as np
import torch

from multi_humanoid_env import MultiHumanoidEnv
from trpo import TRPO, compute_gae, device


def collect_rollout(env, policy, value_net, steps, gamma, lam):
    """
    Collect `steps` timesteps from the shared sim (steps * n_agents total
    transitions), batching the policy/value forward pass across all agents
    at every timestep -- one forward pass covers the whole sim, same as the
    AsyncVectorEnv version's batching, just without the IPC round-trip.

    If env.render_mode == "human", env.step() will sync the live viewer
    once per call automatically (handled inside MultiHumanoidEnv.step).
    """
    n_agents = env.n_agents
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    obs_buf = np.zeros((steps, n_agents, obs_dim), dtype=np.float32)
    act_buf = np.zeros((steps, n_agents, act_dim), dtype=np.float32)
    rew_buf = np.zeros((steps, n_agents), dtype=np.float32)
    done_buf = np.zeros((steps, n_agents), dtype=np.float32)
    val_buf = np.zeros((steps, n_agents), dtype=np.float32)
    logp_buf = np.zeros((steps, n_agents), dtype=np.float32)
    mu_buf = np.zeros((steps, n_agents, act_dim), dtype=np.float32)
    std_buf = np.zeros((steps, n_agents, act_dim), dtype=np.float32)

    ep_returns, ep_lens = [], []
    running_ret = np.zeros(n_agents, dtype=np.float32)
    running_len = np.zeros(n_agents, dtype=np.int64)

    obs, _ = env.reset()

    policy.eval()
    value_net.eval()

    for t in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)  # (n_agents, obs_dim), CPU
        with torch.no_grad():
            dist = policy(obs_t)
            act_t = dist.sample()
            logp_t = dist.log_prob(act_t).sum(-1)
            val_t = value_net(obs_t)

        act = act_t.numpy()
        clipped_act = np.clip(act, -1.0, 1.0)  # motors already normalized via ctrlrange

        next_obs, rew, terminated, truncated, _ = env.step(clipped_act)
        done = np.logical_or(terminated, truncated).astype(np.float32)

        obs_buf[t] = obs
        act_buf[t] = act
        rew_buf[t] = rew
        done_buf[t] = done
        val_buf[t] = val_t.numpy()
        logp_buf[t] = logp_t.numpy()
        mu_buf[t] = dist.mean.numpy()
        std_buf[t] = dist.stddev.numpy()

        running_ret += rew
        running_len += 1
        for i in range(n_agents):
            if done[i]:
                ep_returns.append(running_ret[i])
                ep_lens.append(running_len[i])
                running_ret[i] = 0.0
                running_len[i] = 0

        obs = next_obs

    with torch.no_grad():
        last_val = value_net(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    adv_buf = np.zeros_like(rew_buf)
    ret_buf = np.zeros_like(rew_buf)
    for i in range(n_agents):
        adv_i, ret_i = compute_gae(
            rew_buf[:, i], val_buf[:, i], done_buf[:, i], last_val[i],
            gamma=gamma, lam=lam,
        )
        adv_buf[:, i] = adv_i
        ret_buf[:, i] = ret_i

    def flat(x):
        return x.reshape(steps * n_agents, *x.shape[2:])

    batch = dict(
        obs=torch.as_tensor(flat(obs_buf), dtype=torch.float32),
        act=torch.as_tensor(flat(act_buf), dtype=torch.float32),
        adv=torch.as_tensor(flat(adv_buf), dtype=torch.float32),
        ret=torch.as_tensor(flat(ret_buf), dtype=torch.float32),
        logp=torch.as_tensor(flat(logp_buf), dtype=torch.float32),
        mu=torch.as_tensor(flat(mu_buf), dtype=torch.float32),
        std=torch.as_tensor(flat(std_buf), dtype=torch.float32),
    )
    stats = dict(
        mean_return=float(np.mean(ep_returns)) if ep_returns else float("nan"),
        n_episodes=len(ep_returns),
        mean_len=float(np.mean(ep_lens)) if ep_lens else steps,
    )
    return batch, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--steps-per-iter", type=int, default=4000,
                         help="total transitions per TRPO update (across all agents)")
    parser.add_argument("--num-agents", type=int, default=8,
                         help="humanoids sharing the one simulation")
    parser.add_argument("--spacing", type=float, default=3.0,
                         help="grid spacing between agents' spawn points (meters)")
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--render", action="store_true",
                         help="open a live MuJoCo viewer window during training")
    parser.add_argument("--render-every", type=int, default=1,
                         help="only show the live viewer once every N iterations "
                              "(default 1 = every iteration). Ignored if --render is not set.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    steps = max(1, args.steps_per_iter // args.num_agents)
    print(f"Using 1 sim x {args.num_agents} agents x {steps} steps "
          f"= {args.num_agents * steps} transitions/iter")

    # Two envs are created when using --render-every > 1: a fast headless env
    # for most iterations, and a render-enabled env reused only on the
    # iterations we actually want to watch. Creating a fresh MultiHumanoidEnv
    # per render toggle would rebuild the MJCF/model every time, so instead
    # we keep one headless env and one rendering env alive simultaneously
    # when --render is set, and pick which one to roll out with each iter.
    headless_env = MultiHumanoidEnv(n_agents=args.num_agents, spacing=args.spacing,
                                     render_mode="human")
    render_env = None
    if args.render:
        render_env = MultiHumanoidEnv(n_agents=args.num_agents, spacing=args.spacing,
                                       render_mode="human")

    obs_dim = headless_env.single_observation_space.shape[0]
    act_dim = headless_env.single_action_space.shape[0]

    agent = TRPO(
        obs_dim, act_dim,
        max_kl=args.max_kl, damping=args.damping,
        gamma=args.gamma, lam=args.lam,
    )
    agent.policy.to("cpu")
    agent.value.to("cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        use_render = args.render and (it % args.render_every == 0) and render_env is not None
        env = render_env if use_render else headless_env

        if use_render and not env._renderer_is_running():
            # Viewer window was closed by the user; fall back to headless
            # for the rest of training instead of erroring out.
            print("Viewer window closed; continuing training headless.")
            render_env = None
            env = headless_env

        batch, stats = collect_rollout(env, agent.policy, agent.value, steps, args.gamma, args.lam)

        agent.policy.to(device)
        agent.value.to(device)
        info = agent.update(
            batch["obs"], batch["act"], batch["adv"], batch["ret"],
            batch["logp"], batch["mu"], batch["std"],
        )
        agent.policy.to("cpu")
        agent.value.to("cpu")

        elapsed = time.time() - t0
        print(
            f"iter {it:4d} | return {stats['mean_return']:8.2f} | "
            f"episodes {stats['n_episodes']:3d} | KL {info['kl']:.4f} | "
            f"vloss {info['value_loss']:8.3f} | accepted {info['step_accepted']} | "
            f"{'[rendered] ' if use_render else ''}time {elapsed:6.1f}s"
        )

        if it % args.save_every == 0:
            torch.save(agent.policy.state_dict(), os.path.join(args.out_dir, f"policy_{it}.pt"))

    torch.save(agent.policy.state_dict(), os.path.join(args.out_dir, "policy_final.pt"))
    print("Training complete.")
    headless_env.close()
    if render_env is not None:
        render_env.close()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# To get BOTH multi-core parallelism AND multiple robots per sim: wrap
# MultiHumanoidEnv in gymnasium's AsyncVectorEnv, e.g. K=4 processes each
# running M=8 agents = 32 agents total, spread across 4 cores:
#
#   from gymnasium.vector import AsyncVectorEnv
#   def _env_fn(): return MultiHumanoidEnv(n_agents=8)
#   vec_env = AsyncVectorEnv([_env_fn for _ in range(4)])
#   # vec_env.step() then takes/returns arrays of shape (4, 8, act_dim) flattened
#   # to (32, act_dim) by AsyncVectorEnv's own batching of single_action_space.
#
# You'd set MultiHumanoidEnv.single_action_space / single_observation_space
# to the *per-agent* (not per-sim) shapes for this to compose correctly with
# AsyncVectorEnv's assumptions -- or just treat each 8-agent sim as reporting
# an (8, act_dim) action space directly, since AsyncVectorEnv doesn't require
# scalar per-env action spaces.
#
# Note: render_env and headless_env above are two independent MjModel/MjData
# instances (same XML, separately loaded), so switching between them mid-run
# is safe -- they don't share physics state, only the trained policy/value
# networks are shared across both.
# ---------------------------------------------------------------------------