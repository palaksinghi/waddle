import os
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walker2d.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class CustomWalker_2dEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        frame_skip=5,
        healthy_reward=1.0,
        ctrl_cost=1e-3,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        **kwargs,
    ):
        # 1. Properly pass parameters to EzPickle for vector env serialization
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            frame_skip,
            healthy_reward,
            ctrl_cost,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            **kwargs,
        )

        self._forward_reward = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._ctrl_cost = ctrl_cost
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale

        # Observation size for Walker2d: 8 (qpos minus x) + 9 (qvel) = 17
        obs_size = 17
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # 2. Pass frame_skip parameter correctly
        MujocoEnv.__init__(
            self,
            XML_PATH,
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _get_obs(self):
        qpos = self.data.qpos.flatten()[1:]
        qvel = self.data.qvel.flatten()
        return np.concatenate([qpos, qvel])

    def is_healthy(self):
        z, angle = self.data.qpos[1], self.data.qpos[2]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        return (min_z < z < max_z) and (min_angle < angle < max_angle)

    def control_cost(self, action):
        return self._ctrl_cost * np.sum(np.square(action))

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        vel = self.data.qvel[0]
        forward_reward = self._forward_reward * vel
        ctrl_cost = self.control_cost(action)

        # 3. Compute healthy status once per step
        healthy = self.is_healthy()
        h_reward = self._healthy_reward if healthy else 0.0

        reward = forward_reward + h_reward - ctrl_cost
        terminated = not healthy
        truncated = False
        obs = self._get_obs()

        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_alive": h_reward,
            "x_velocity": vel,
        }

        # Gymnasium handles rendering automatically based on render_mode
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        q_pos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        q_vel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(q_pos, q_vel)
        return self._get_obs()


# Register custom env with Gymnasium
gym.register(
    id="CustomWalker2d-v0",
    entry_point=__name__ + ":CustomWalker_2dEnv",
    max_episode_steps=1000,
)


def make_env(env_id="CustomWalker2d-v0", seed=41, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk