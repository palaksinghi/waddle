"""
Play/evaluate a trained open_duck_mini_v2 policy in a small windowed env.

Usage:
    python scripts/play.py --load_run <run_name> --checkpoint model_3000.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained open_duck_mini_v2 policy.")
parser.add_argument("--task", type=str, default="Isaac-OpenDuck-Flat-Play-v0")
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from open_duck_mini_v2.open_duck_flat_env_cfg import OpenDuckFlatEnvCfg_PLAY
from scripts.train import make_ppo_runner_cfg


def main():
    env_cfg = OpenDuckFlatEnvCfg_PLAY()
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = make_ppo_runner_cfg()
    checkpoint_path = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name,
                                    args_cli.load_run, args_cli.checkpoint)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()