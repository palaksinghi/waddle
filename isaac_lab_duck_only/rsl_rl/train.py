"""
Train script for open_duck_mini_v2 flat-ground gait using rsl_rl (PPO).

Usage:
    python scripts/train.py --num_envs 4096 --headless
    python scripts/train.py --num_envs 64                 # windowed, for visual debugging
    python scripts/train.py --resume --load_run <run_name> --checkpoint <ckpt>

This follows the standard IsaacLab + rsl_rl training entrypoint pattern.
Adjust `--task` to whatever you register the env as in your gym registry
(see note at bottom of this file re: registering the task).
"""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI args -- parsed BEFORE launching the sim app (IsaacLab requirement)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train open_duck_mini_v2 gait with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Isaac-OpenDuck-Flat-v0", help="Gym-registered task name.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=3000, help="RL training iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to resume from.")

# adds --headless, --device, --enable_cameras, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything below imports isaaclab/isaacsim modules -- must happen AFTER
# AppLauncher has started the sim app.
# ---------------------------------------------------------------------------
import gymnasium as gym
import os
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from env.open_duck_flat_env_cfg import OpenDuckFlatEnvCfg


def make_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO hyperparameters. Reasonable starting point for a small biped --
    retune entropy_coef / learning_rate if training is unstable or plateaus.
    """
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=args_cli.max_iterations,
        save_interval=100,
        experiment_name="open_duck_mini_v2_flat",
        empirical_normalization=True,
        policy=dict(
            class_name="ActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=dict(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def main():
    env_cfg: ManagerBasedRLEnvCfg = OpenDuckFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    env = gym.make(args_cli.task, cfg=env_cfg)  # requires task registration, see note below
    env = RslRlVecEnvWrapper(env)

    runner_cfg = make_ppo_runner_cfg()
    log_dir = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=env_cfg.sim.device)

    if args_cli.resume:
        assert args_cli.load_run is not None and args_cli.checkpoint is not None, \
            "--load_run and --checkpoint are required when --resume is set"
        resume_path = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name,
                                    args_cli.load_run, args_cli.checkpoint)
        runner.load(resume_path)
        print(f"[INFO] Resumed from checkpoint: {resume_path}")

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    torch.manual_seed(args_cli.seed)
    main()
    simulation_app.close()

# ---------------------------------------------------------------------------
# NOTE on task registration:
# gym.make() above needs the task registered first. Add this to your
# open_duck_mini_v2/__init__.py (package root, not the mdp submodule):
#
#   import gymnasium as gym
#   from .open_duck_flat_env_cfg import OpenDuckFlatEnvCfg, OpenDuckFlatEnvCfg_PLAY
#
#   gym.register(
#       id="Isaac-OpenDuck-Flat-v0",
#       entry_point="isaaclab.envs:ManagerBasedRLEnv",
#       kwargs={"env_cfg_entry_point": OpenDuckFlatEnvCfg},
#   )
#   gym.register(
#       id="Isaac-OpenDuck-Flat-Play-v0",
#       entry_point="isaaclab.envs:ManagerBasedRLEnv",
#       kwargs={"env_cfg_entry_point": OpenDuckFlatEnvCfg_PLAY},
#   )
# ---------------------------------------------------------------------------

