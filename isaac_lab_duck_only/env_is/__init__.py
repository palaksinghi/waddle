import gymnasium as gym

from .duck_gait_env import DuckGaitEnv
from .duck_gait_env_cfg import DuckGaitEnvCfg

gym.register(
    id="Isaac-OpenDuckMiniV2-FlatGait-v0",
    entry_point="env.duck_gait_env:DuckGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DuckGaitEnvCfg,
        "rsl_rl_cfg_entry_point": "rl.rsl_rl_ppo_cfg:DuckGaitPPORunnerCfg",
    },
)