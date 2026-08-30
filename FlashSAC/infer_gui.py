import os
os.environ["OMP_NUM_THREADS"] = "2"

import time
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.open_duck_mini_v2 import OpenDuckBipedalEnv

CHECKPOINT_PATH = "models/open_duck/open_duck_mini_v2/OpenDuckMiniV2-v0/seed0-0829-124551/step10000"
CONFIG_PATH = "./configs"
CONFIG_NAME = "flashSAC_base"


def main():
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[])
    OmegaConf.resolve(cfg)

    env = OpenDuckBipedalEnv(render_mode="human")

    _, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(CHECKPOINT_PATH)

    obs, _ = env.reset()
    transition = None
    step = 0
    while True:
        if transition is not None:
            # add a batch dimension: network (incl. BatchNorm) expects (num_envs, obs_dim)
            batched_transition = {k: (np.expand_dims(v, 0) if isinstance(v, np.ndarray) else v)
                                   for k, v in transition.items()}
            action = agent.sample_actions(step, prev_transition=batched_transition, training=False)
            action = np.array(action)[0]  # remove batch dimension
        else:
            action = env.action_space.sample()
        action = np.array(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        transition = {
            "observation": obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_observation": next_obs,
        }
        obs = next_obs
        env.render()
        time.sleep(env.dt)  # real-time playback

        step += 1
        if terminated or truncated:
            obs, _ = env.reset()
            transition = None


if __name__ == "__main__":
    main()