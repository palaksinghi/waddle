"""Custom MuJoCo HalfCheetah environment with an explicit reward definition."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

DEFAULT_XML = Path(__file__).with_name("half_cheetah.xml")


class HalfCheetahEnv(MujocoEnv):
    # MuJoCo step = 5 simulation frames × 0.01 s = 0.05 s (20 FPS).
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self, xml_file: str | Path = DEFAULT_XML, forward_reward_weight: float = 1.0,
        healthy_reward: float = 1.0, ctrl_cost_weight: float = 0.1,
        healthy_z_range: tuple[float, float] = (0.25, 1.0),
        healthy_angle_range: tuple[float, float] = (-1.0, 1.0),
        terminate_when_unhealthy: bool = True, render_mode: str | None = None,
    ):
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward_value = healthy_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range
        self.terminate_when_unhealthy = terminate_when_unhealthy
        observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float64)
        super().__init__(str(xml_file), frame_skip=5, observation_space=observation_space, render_mode=render_mode)
        self._torso_id = self.model.body("torso").id

    @property
    def is_healthy(self) -> bool:
        # qpos[rootz] is relative to the torso body's 0.7 m XML offset;
        # xpos is the actual world-frame torso height used for health checks.
        torso_z, torso_angle = float(self.data.xpos[self._torso_id, 2]), float(self.data.qpos[2])
        return self.healthy_z_range[0] <= torso_z <= self.healthy_z_range[1] and self.healthy_angle_range[0] <= torso_angle <= self.healthy_angle_range[1]

    def _get_obs(self) -> np.ndarray:
        # Exclude root x position so observations are translation invariant.
        return np.concatenate((self.data.qpos.flat[1:], self.data.qvel.flat)).astype(np.float32)

    def step(self, action: np.ndarray):
        x_before = float(self.data.qpos[0])
        self.do_simulation(action, self.frame_skip)
        x_velocity = (float(self.data.qpos[0]) - x_before) / self.dt

        forward_reward = self.forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward_value if self.is_healthy else 0.0
        control_penalty = self.ctrl_cost_weight * float(np.square(action).sum())
        reward = forward_reward + healthy_reward - control_penalty

        terminated = self.terminate_when_unhealthy and not self.is_healthy
        info = {"x_velocity": x_velocity, "reward_forward": forward_reward,
                "reward_healthy": healthy_reward, "reward_ctrl_penalty": -control_penalty}
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(0.0, 0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
