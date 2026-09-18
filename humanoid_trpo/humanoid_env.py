"""
Thin Gymnasium MuJoCo environment around our custom humanoid.xml.
Reward mirrors the standard Gymnasium Humanoid-v4 shaping:
  forward velocity + alive bonus - control cost - contact cost,
  with early termination if the torso falls outside a healthy z-range.
"""

import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Humanoid.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class CustomHumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        obs_size = 45 + 23 * 6  # qpos(-2) + qvel + cinert + cvel + qfrc_actuator + cfrc_ext (approx, auto-fixed below)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            XML_PATH,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        # observation space depends on model dims discovered at load time; fix it up now
        obs = self._get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        qpos = self.data.qpos.flatten()[2:]        # drop x, y (translation-invariant)
        qvel = self.data.qvel.flatten()
        cinert = self.data.cinert.flatten()
        cvel = self.data.cvel.flatten()
        qfrc_actuator = self.data.qfrc_actuator.flatten()
        cfrc_ext = self.data.cfrc_ext.flatten()
        return np.concatenate([qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext])

    @property
    def healthy_reward(self):
        return self._healthy_reward if self.is_healthy else 0.0

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        return min_z < z < max_z

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def contact_cost(self):
        cost = self._contact_cost_weight * np.sum(np.square(self.data.cfrc_ext))
        lo, hi = self._contact_cost_range
        return np.clip(cost, lo, hi)

    def step(self, action):
        xy_before = self.data.qpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.qpos[:2].copy()

        vel = (xy_after - xy_before) / self.dt
        forward_reward = self._forward_reward_weight * vel[0]

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        reward = forward_reward + self.healthy_reward - ctrl_cost - contact_cost

        terminated = not self.is_healthy
        obs = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_alive": self.healthy_reward,
            "x_velocity": vel[0],
            "y_velocity": vel[1],
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


def make_env(render_mode=None):
    return CustomHumanoidEnv(render_mode="human")