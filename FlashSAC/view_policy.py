import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import sys

import hydra
import numpy as np
from dm_control import suite, viewer
from gymnasium.spaces.utils import flatten
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.dmc import make_dmc_env


def run(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))

    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    domain_name, task_name = cfg.env.env_name.split("-")

    # build a throwaway flattened gym env, purely to get correctly-shaped
    # observation_space / action_space / env_info for agent construction,
    # matching exactly what train.py used.
    flat_gym_env = make_dmc_env(cfg.env.env_name, seed=cfg.seed, flatten=True)
    observation_space = flat_gym_env.observation_space
    action_space = flat_gym_env.action_space
    _, env_info = flat_gym_env.reset()

    # unflattened env, just to grab the Dict observation_space for flatten()
    dict_gym_env = make_dmc_env(cfg.env.env_name, seed=cfg.seed, flatten=False)
    dict_obs_space = dict_gym_env.observation_space
    dict_gym_env.close()
    flat_gym_env.close()

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

    # raw dm_control env, used only for physics + the live viewer window
    raw_env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": cfg.seed},
    )

    transition_holder = {"prev_transition": None}

    def policy(time_step):
        obs_flat = flatten(dict_obs_space, time_step.observation).astype(np.float32)
        obs_batched = obs_flat[None, :]  # add fake batch dim of 1

        prev_transition = {"next_observation": obs_batched}
        actions = agent.sample_actions(
            interaction_step=0,
            prev_transition=prev_transition,
            training=False,
        )
        actions = np.array(actions)[0]  # remove batch dim
        return actions

    print("Launching interactive MuJoCo viewer... close the window to exit.")
    viewer.launch(raw_env, policy=policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", type=str, action="append", default=[])
    args = parser.parse_args()
    run(args)
