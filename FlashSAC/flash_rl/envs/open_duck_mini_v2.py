# flash_rl/envs/open_duck_mini_v2.py
#
# This is your corrected env file (bugs fixed per review) PLUS a
# gymnasium.register() call at the bottom.
#
# IMPORTANT: I don't have visibility into flash_rl/envs/__init__.py or
# however the repo's mujoco env factory actually discovers custom tasks.
# grep flash_rl/envs/ locally for the pattern used by existing MuJoCo
# tasks (e.g. how HalfCheetah / Ant get resolved from env_name) and match
# that pattern instead of this register() call if it differs.

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import mujoco.viewer
from . import rewardsm as R

XML_PATH = "robot/open_duck_mini_v2/scene.xml"
GROUND_GEOM = "floor"
TARGET_HEIGHT = 0.17
GAIT_PERIOD = 0.5
CONTACT_FORCE_THRESHOLD = 1.0
BAD_TILT_LIMIT = np.deg2rad(45)
MAX_EPISODE_SECONDS = 40.0
TARGET_FEET_AIR_TIME = GAIT_PERIOD / 2.0


class OpenDuckBipedalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path: str = XML_PATH, render_mode=None, debug: bool = False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt_control = self.model.opt.timestep
        self.render_mode = render_mode
        self.debug = debug  # FIX: gate the per-step debug print behind a flag
        self.viewer = None

        # FIX: gait-symmetry pose buffers are declared but were never
        # populated anywhere, silently zeroing the "symmetry" reward term.
        # Buffer the *previous half-gait-cycle* leg pose so symmetry_penalty
        # has something real to compare against.
        self.left_leg_pose_buffer = None
        self.right_leg_pose_buffer = None
        self._half_period_steps = None  # set in reset(), needs self.dt

        self.frame_skip = max(1, int(round((1.0 / 50) / self.model.opt.timestep)))
        self.dt = self.model.opt.timestep * self.frame_skip

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.joint_lower = self.model.jnt_range[:, 0][-self.nu:]
        self.joint_upper = self.model.jnt_range[:, 1][-self.nu:]

        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_assembly_2")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_assembly")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
        self.default_leg_joint_pos = np.zeros(self.nu, dtype=np.float32)
        self.prev_leg_joint_vel = np.zeros(self.nu, dtype=np.float32)

        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_prev_action = np.zeros_like(self.prev_action)
        self.prev_base_pos_xy = np.zeros(2, dtype=np.float32)
        self.last_air_time = np.zeros(2, dtype=np.float32)
        self.was_in_contact = np.zeros(2, dtype=bool)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.episode_t = 0.0
        self.initial_yaw = 0.0
        self.spawn_yaw = 0.0
        self.spawn_xy = np.zeros(2, dtype=np.float32)
        self._steps_since_half_cycle = 0

    def _projected_gravity(self):
        quat = self.data.qpos[3:7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        mat = mat.reshape(3, 3)
        g_world = np.array([0, 0, -1.0])
        return mat.T @ g_world

    def _root_lin_vel_b(self):
        quat = self.data.qpos[3:7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        mat = mat.reshape(3, 3)
        v_world = self.data.qvel[0:3]
        return mat.T @ v_world

    def _root_ang_vel_b(self):
        return self.data.qvel[3:6]

    def _obs_dim(self):
        return 2 + 3 + 3 + self.nu + self.nu + 3

    def _get_obs(self):
        phase = R.phase_vector(self.episode_t, GAIT_PERIOD)
        lin_vel_b = self._root_lin_vel_b()
        ang_vel_b = self._root_ang_vel_b()
        joint_pos = self.data.qpos[-self.nu:]
        joint_vel = self.data.qvel[-self.nu:]
        obs = np.concatenate([phase, lin_vel_b, ang_vel_b, joint_pos, joint_vel, self.cmd]).astype(np.float32)
        return obs

    def _foot_contact(self, foot_body_id):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[c.geom1]
            geom2_body = self.model.geom_bodyid[c.geom2]
            if (geom1_body == foot_body_id and c.geom2 == self.ground_id) or \
               (geom2_body == foot_body_id and c.geom1 == self.ground_id):
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if np.linalg.norm(force[:3]) > CONTACT_FORCE_THRESHOLD:
                    return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # NOTE: removed the no-op `else` branch (self.data.qpos[:] = self.data.qpos.copy()).
        # If your scene.xml has no <keyframe>, mj_resetData already leaves qpos at the
        # model's qpos0 default -- decide explicitly here if you need a different spawn pose.

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.initial_yaw = R.quat_to_yaw(self.data.qpos[3:7])
        self.spawn_yaw = self.initial_yaw
        self.spawn_xy = self.data.qpos[0:2].copy()
        self.prev_base_pos_xy = self.data.qpos[0:2].copy()
        self.prev_action[:] = 0
        self.prev_prev_action[:] = 0
        self.last_air_time[:] = 0
        self.was_in_contact[:] = False
        self.episode_t = 0.0
        self._steps_since_half_cycle = 0
        self.left_leg_pose_buffer = None
        self.right_leg_pose_buffer = None
        self._half_period_steps = max(1, int(round((GAIT_PERIOD / 2.0) / self.dt)))

        FORWARD_VEL = 0.2
        self.cmd = np.zeros(3, dtype=np.float32)
        self.cmd[0] = FORWARD_VEL
        self.cmd[1] = 0.0
        self.cmd[2] = 0.0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        lo, hi = self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        ctrl = lo + (action + 1.0) * 0.5 * (hi - lo)
        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.episode_t += self.dt

        left_contact = self._foot_contact(self.left_foot_body_id)
        right_contact = self._foot_contact(self.right_foot_body_id)

        first_contact = np.array([
            left_contact and not self.was_in_contact[0],
            right_contact and not self.was_in_contact[1],
        ])
        air_time_snapshot = self.last_air_time.copy()
        self.last_air_time[0] = 0.0 if left_contact else self.last_air_time[0] + self.dt
        self.last_air_time[1] = 0.0 if right_contact else self.last_air_time[1] + self.dt
        self.was_in_contact[0], self.was_in_contact[1] = left_contact, right_contact

        # FIX: actually populate the leg-pose buffers every half gait cycle
        # so symmetry_penalty is no longer a permanent no-op.
        self._steps_since_half_cycle += 1
        if self._steps_since_half_cycle >= self._half_period_steps:
            leg_joint_pos = self.data.qpos[-self.nu:].copy()
            self.left_leg_pose_buffer = leg_joint_pos[0:5].copy()
            self.right_leg_pose_buffer = leg_joint_pos[5:10].copy()
            self._steps_since_half_cycle = 0

        gravity = self._projected_gravity()
        terminated = R.bad_orientation(gravity, BAD_TILT_LIMIT)
        truncated = self.episode_t >= MAX_EPISODE_SECONDS

        reward, info = self._compute_reward(
            action, left_contact, right_contact, first_contact, air_time_snapshot, gravity, terminated
        )

        if self.debug:
            w = R.REWARD_WEIGHTS
            weighted = {k: round(w[k] * v, 3) for k, v in info["reward_terms"].items()}
            print(f"step_reward={reward:.3f}  {weighted}")

        info["terminated"] = bool(terminated)
        if truncated:
            info["TimeLimit.truncated"] = True

        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()
        self.prev_base_pos_xy = self.data.qpos[0:2].copy()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action, left_contact, right_contact, first_contact, air_time_snapshot, gravity, terminated):
        w = R.REWARD_WEIGHTS
        lin_vel_b = self._root_lin_vel_b()
        ang_vel_b = self._root_ang_vel_b()
        joint_pos = self.data.qpos[-self.nu:]
        phase_angle = 2.0 * np.pi * ((self.episode_t % GAIT_PERIOD) / GAIT_PERIOD)
        phase_left = phase_angle
        phase_right = phase_angle + np.pi
        left_foot_pos = self.data.xpos[self.left_foot_body_id].copy()
        right_foot_pos = self.data.xpos[self.right_foot_body_id].copy()
        leg_joint_pos = self.data.qpos[-self.nu:].copy()
        leg_joint_vel = self.data.qvel[-self.nu:].copy()
        actuator_force = self.data.actuator_force.copy()
        feet_z = min(left_foot_pos[2], right_foot_pos[2])
        base_height_rel = self.data.qpos[2] - feet_z

        terms = {
            "track_lin_vel_xy_exp": R.track_lin_vel_xy_exp(lin_vel_b[:2], self.cmd[:2], std=0.5),
            "track_ang_vel_z_exp": R.track_ang_vel_z_exp(ang_vel_b[2], self.cmd[2], std=0.5),
            "forward_progress": R.forward_progress(self.data.qpos[0:2], self.prev_base_pos_xy),
            "heading_drift": R.heading_drift_penalty(R.quat_to_yaw(self.data.qpos[3:7]), self.spawn_yaw),
            "lateral_path_deviation": R.lateral_path_deviation_penalty(self.data.qpos[0:2], self.spawn_xy, self.spawn_yaw),
            "yaw_penalty": R.yaw_penalty(ang_vel_b[2], self.cmd[2]),
            "gait_phase_tracking": R.gait_phase_tracking_reward(phase_left, phase_right, float(left_contact), float(right_contact)),
            "feet_air_time_reward": R.feet_air_time_reward(first_contact, air_time_snapshot, TARGET_FEET_AIR_TIME, self.cmd[:2]),
            "symmetry": R.symmetry_penalty(leg_joint_pos, self.left_leg_pose_buffer, self.right_leg_pose_buffer),
            "flat_orientation_l2": R.flat_orientation_l2(gravity),
            "base_height_l2": R.base_height_l2(base_height_rel, TARGET_HEIGHT),
            "pelvis_vel_tracking": R.pelvis_vel_tracking_penalty(lin_vel_b[:2], self.cmd[:2]),
            "lateral_spread": R.lateral_spread_penalty(left_foot_pos, right_foot_pos),
            "gait_phase_contact": R.gait_phase_contact_reward(self.episode_t, GAIT_PERIOD, left_contact, right_contact),
            "joint_pos_limits": R.joint_pos_limits(joint_pos, self.joint_lower, self.joint_upper),
            "joint_penalty": R.joint_penalty(leg_joint_pos, self.default_leg_joint_pos),
            "joint_vel": R.joint_vel_penalty(leg_joint_vel),
            "joint_acc": R.joint_acc_penalty(leg_joint_vel, self.prev_leg_joint_vel, self.dt_control),
            "torque": R.torque_penalty(actuator_force, leg_joint_vel),
            "action_rate_l2": R.action_rate_l2(action, self.prev_action),
            "action_smoothness2_l2": R.action_smoothness2_l2(action, self.prev_action, self.prev_prev_action),
            "lin_vel_z_l2": R.lin_vel_z_l2(lin_vel_b[2]),          # FIX: now actually wired in (weight already existed)
            "ang_vel_xy_l2": R.ang_vel_xy_l2(ang_vel_b),           # FIX: now actually wired in (weight already existed)
            "alive_cost": R.alive_cost(),
            "is_terminated": float(terminated),
        }
        self.prev_leg_joint_vel = leg_joint_vel.copy()  # FIX: this was never updated, so joint_acc was always wrong
        total = sum(w[k] * v for k, v in terms.items())
        return float(total), {"reward_terms": terms}

    def render(self):
        if self.render_mode == "rgb_array":
            if not hasattr(self, "_rgb_renderer") or self._rgb_renderer is None:
                self._rgb_renderer = mujoco.Renderer(self.model, height=480, width=480)
            self._rgb_renderer.update_scene(self.data)
            return self._rgb_renderer.render()  # returns (H, W, 3) uint8 array

        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None

        else:
            return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if hasattr(self, "_rgb_renderer") and self._rgb_renderer is not None:
            self._rgb_renderer.close()
            self._rgb_renderer = None


register(
    id="OpenDuckMiniV2-v0",
    entry_point="flash_rl.envs.open_duck_mini_v2:OpenDuckBipedalEnv",
    max_episode_steps=int(MAX_EPISODE_SECONDS * 50),
)
