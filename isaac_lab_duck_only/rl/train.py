"""Train open_duck_mini_v2 flat-ground gait with PPO via rsl_rl.

Run from ~/waddle/IsaacLab/:
    ./isaaclab.sh -p scripts/rl/train.py --num_envs 16 --headless
"""

import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI args - AppLauncher must be constructed before any isaaclab.* imports
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train open_duck_mini_v2 flat gait (PPO / rsl_rl).")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from --load_run/--checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from (under logs dir).")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument("--run_name", type=str, default=None, help="Suffix for this run's log folder.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Everything below needs the sim app already running
# -----------------------------------------------------------------------------
import os
import sys
from datetime import datetime

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import env  # noqa: F401  (registers Isaac-OpenDuckMiniV2-FlatGait-v0 via gym.register in env/__init__.py)
from env.duck_gait_env_cfg import DuckGaitEnvCfg
from rl.rsl_rl_ppo_cfg import DuckGaitPPORunnerCfg

TASK_ID = "Isaac-OpenDuckMiniV2-FlatGait-v0"

# Keys that older rsl_rl checkpoints/cfgs may still carry but current
# OnPolicyRunner no longer accepts. Stripped defensively wherever present.
_DEPRECATED_MODEL_KEYS = ["stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"]


def _strip_deprecated_keys(agent_dict: dict) -> dict:
    """Remove deprecated keys from actor/critic (or policy) sub-configs, if present."""
    for section_name in ("actor", "critic", "policy"):
        section = agent_dict.get(section_name)
        if isinstance(section, dict):
            for key in _DEPRECATED_MODEL_KEYS:
                section.pop(key, None)
    return agent_dict


def main():
    env_cfg: ManagerBasedRLEnvCfg = DuckGaitEnvCfg()
    agent_cfg: RslRlOnPolicyRunnerCfg = DuckGaitPPORunnerCfg()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        run_name += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    dump_yaml(os.path.join(log_dir, "env_cfg.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "agent_cfg.yaml"), agent_cfg)

    gym_env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    gym_env = RslRlVecEnvWrapper(gym_env)

    agent_dict = _strip_deprecated_keys(agent_cfg.to_dict())
    runner = OnPolicyRunner(gym_env, agent_dict, log_dir=log_dir, device=agent_cfg.device)

    if args_cli.resume:
        if args_cli.checkpoint:
            resume_path = args_cli.checkpoint
        elif args_cli.load_run:
            run_dir = os.path.join(log_root, args_cli.load_run)
            if not os.path.isdir(run_dir):
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
            ckpts = sorted(f for f in os.listdir(run_dir) if f.startswith("model_"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {run_dir}")
            resume_path = os.path.join(run_dir, ckpts[-1])
        else:
            raise ValueError("--resume requires --checkpoint or --load_run")
        print(f"[INFO] Resuming training from checkpoint: {resume_path}")
        runner.load(resume_path)

    print_dict(agent_dict, nesting=0)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()