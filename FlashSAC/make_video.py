import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import sys

import hydra
import imageio
import numpy as np
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs import create_envs
from flash_rl.evaluation import record_video


def run(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))

    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    train_env, eval_env, record_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    assert cfg.agent_load_path is not None, "must pass agent_load_path override"
    load_path = os.path.join(script_dir, cfg.agent_load_path)
    print(f"Loading agent from: {load_path}")
    agent.load(load_path)

    print("Recording rollout...")
    video_info = record_video(agent, record_env, args.num_episodes, cfg.env.env_type)
    video = video_info["video"]  # (b, t, c, h, w)
    print(f"Got video array with shape: {video.shape}")

    # take first episode, convert (t, c, h, w) -> (t, h, w, c), uint8
    clip = video[0]
    clip = np.transpose(clip, (0, 2, 3, 1))
    if clip.dtype != np.uint8:
        clip = np.clip(clip, 0, 255).astype(np.uint8)

    out_path = args.out
    print(f"Writing video to: {out_path}")
    imageio.mimsave(out_path, list(clip), fps=args.fps)
    print("Done.")

    train_env.close()
    eval_env.close()
    record_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default="walk.mp4")
    args = parser.parse_args()
    run(args)
