"""Play (evaluate / visualize) a trained open_duck_mini_v2 flat-gait policy.

Run from ~/waddle/IsaacLab/:
    ./isaaclab.sh -p /absolute/path/to/rl/play.py --load_run 2026-08-20_12-00-00 --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained open_duck_mini_v2 flat-gait policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of envs to visualize.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder under logs dir to load from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file; defaults to latest.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video instead of live rendering.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import env  # noqa: F401  (registers Isaac-OpenDuckMiniV2-FlatGait-v0)
from env.duck_gait_env_cfg import DuckGaitEnvCfg
from rl.rsl_rl_ppo_cfg import DuckGaitPPORunnerCfg

TASK_ID = "Isaac-OpenDuckMiniV2-FlatGait-v0"


def find_checkpoint(run_dir: str, explicit: str | None) -> str:
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(run_dir, explicit)
    ckpts = sorted(f for f in os.listdir(run_dir) if f.startswith("model_"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return os.path.join(run_dir, ckpts[-1])


def main():
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", DuckGaitPPORunnerCfg().experiment_name))
    run_dir = os.path.join(log_root, args_cli.load_run)

    # prefer the exact cfg the run was trained with, fall back to current code's cfg
    env_cfg_path = os.path.join(run_dir, "env_cfg.pkl")
    env_cfg = DuckGaitEnvCfg()
    agent_cfg_path = os.path.join(run_dir, "agent_cfg.pkl")
    agent_cfg = DuckGaitPPORunnerCfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = agent_cfg.device if hasattr(agent_cfg, "device") else "cuda:0"

    render_mode = "rgb_array" if args_cli.video else None
    gym_env = gym.make(TASK_ID, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = os.path.join(run_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="play",
        )

    gym_env = RslRlVecEnvWrapper(gym_env)

    checkpoint_path = find_checkpoint(run_dir, args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    _agent_dict = agent_cfg.to_dict()
    _deprecated_model_keys = ["stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"]
    for _k in _deprecated_model_keys:
        _agent_dict["actor"].pop(_k, None)
        _agent_dict["critic"].pop(_k, None)
    runner = OnPolicyRunner(gym_env, _agent_dict, log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=gym_env.unwrapped.device)

    obs = gym_env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = gym_env.step(actions)
        if args_cli.video:
            break  # RecordVideo wrapper stops itself after video_length steps

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()