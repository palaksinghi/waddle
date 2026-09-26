import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import sys
import time

import hydra
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.duck import make_duck_env
from flash_rl.envs.duck.envm import MAX_EPISODE_SECONDS  # noqa: F401  (horizon parity with train)
from gymnasium.wrappers import TimeLimit


def run(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))

    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    # build the duck env exactly as train.py did, to get matching obs/action spaces
    # (train wraps the env in TimeLimit(2000); the raw env already truncates at
    # 40 s / 0.02 s = 2000 steps, so wrap here for identical horizon handling)
    env = TimeLimit(make_duck_env(cfg.env.env_name, seed=cfg.seed), max_episode_steps=2000)
    observation_space = env.observation_space
    action_space = env.action_space
    obs, env_info = env.reset()

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

    model = env.unwrapped.model
    data = env.unwrapped.data

    print("Launching interactive MuJoCo viewer... close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            obs_batched = np.asarray(obs, dtype=np.float32)[None, :]
            prev_transition = {"next_observation": obs_batched}
            actions = agent.sample_actions(
                interaction_step=0,
                prev_transition=prev_transition,
                training=False,
            )
            actions = np.array(actions)[0]

            obs, reward, terminated, truncated, info = env.step(actions)
            viewer.sync()

            if terminated or truncated:
                obs, env_info = env.reset()

            time.sleep(env.unwrapped.dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", type=str, action="append", default=[])
    args = parser.parse_args()
    run(args)
