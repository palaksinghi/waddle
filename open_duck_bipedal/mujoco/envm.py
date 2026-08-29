#IMPORTS
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer
import rewardsm as R

#PATH FOR XML
XML_PATH = "robots/open_duck_mini_v2/scene.xml"
GROUND_GEOM = "floor"
TARGET_HEIGHT = 0.17
GAIT_PERIOD = 0.5  #0.7#20aug
CONTACT_FORCE_THRESHOLD = 1.0
BAD_TILT_LIMIT = np.deg2rad(45)
MAX_EPISODE_SECONDS = 40.0
TARGET_FEET_AIR_TIME = GAIT_PERIOD / 2.0

class OpenDuckBipedalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path: str = XML_PATH, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt_control = self.model.opt.timestep
        self.render_mode = render_mode
        self.viewer = None
        self.left_leg_pose_buffer = None
        self.right_leg_pose_buffer = None

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
        self.default_leg_joint_pos = np.zeros(
        self.nu,
        dtype=np.float32
        )
        self.prev_leg_joint_vel = np.zeros(self.nu, dtype=np.float32)

        self.undesired_ids = []

        self.prev_action = np.zeros(self.action_space.shape,dtype=np.float32)
        self.prev_prev_action = np.zeros_like(self.prev_action)
        self.prev_base_pos_xy = np.zeros(2, dtype=np.float32)
        self.last_air_time = np.zeros(2, dtype=np.float32)
        self.was_in_contact = np.zeros(2, dtype=bool)
        self.cmd = np.zeros(3, dtype=np.float32)  # vx, vy, wz
        self.episode_t = 0.0
        self.initial_yaw = 0.0
        self.spawn_yaw=0.0
        self.spawn_xy=np.zeros(2,dtype=np.float32)

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
        return self.data.qvel[3:6]  # already body-frame in mujoco free joint

    def _obs_dim(self):
        return 2 + 3 + 3 + self.nu + self.nu + 3  # phase, lin_vel, ang_vel, qpos(actuated), qvel(actuated), cmd

    def _get_obs(self):
        phase = R.phase_vector(self.episode_t, GAIT_PERIOD)
        lin_vel_b = self._root_lin_vel_b()
        ang_vel_b = self._root_ang_vel_b()
        joint_pos = self.data.qpos[-self.nu:]
        joint_vel = self.data.qvel[-self.nu:]
        obs = np.concatenate([
            phase, lin_vel_b, ang_vel_b, joint_pos, joint_vel, self.cmd
        ]).astype(np.float32)
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

    def _undesired_contact_flags(self):
        if not self.undesired_ids:
            return np.zeros(1, dtype=np.float32)
        flags = np.zeros(len(self.undesired_ids), dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for idx, gid in enumerate(self.undesired_ids):
                if (c.geom1 == gid and c.geom2 == self.ground_id) or \
                   (c.geom2 == gid and c.geom1 == self.ground_id):
                    flags[idx] = 1.0
        return flags

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # init_qpos = self.model.key_qpos[0].copy() if self.model.nkey > 0 else self.data.qpos.copy()
        # self.data.qpos[:] = init_qpos
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            self.data.qpos[:] = self.data.qpos.copy()

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.initial_yaw = R.quat_to_yaw(self.data.qpos[3:7])
        self.spawn_yaw = self.initial_yaw
        self.spawn_xy = self.data.qpos[0:2].copy()
        self.prev_base_pos_xy = self.data.qpos[0:2].copy()
        self.prev_action[:] = 0
        self.last_air_time[:] = 0
        self.was_in_contact[:] = False
        self.episode_t = 0.0

        # FIX: previously sampled a random cmd and then immediately overwrote it
        # with a fixed forward velocity -- the random sample was dead code.
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

        gravity = self._projected_gravity()
        terminated = R.bad_orientation(gravity, BAD_TILT_LIMIT)
        truncated = self.episode_t >= MAX_EPISODE_SECONDS

        reward, info = self._compute_reward(
            action, left_contact, right_contact, first_contact, air_time_snapshot, gravity, terminated
        )
        ####################################################n
        w = R.REWARD_WEIGHTS
        weighted = {k: round(w[k] * v, 3) for k, v in info["reward_terms"].items()}
        print(f"step_reward={reward:.3f}  {weighted}")
    ######################################################################N
        # FIX: inference script reads info["terminated"] / info["TimeLimit.truncated"]
        # but these were never populated, so fall/timeout status was always wrong.
        info["terminated"] = bool(terminated)
        if truncated:
            info["TimeLimit.truncated"] = True

        self.prev_action = action.copy()
        self.prev_base_pos_xy = self.data.qpos[0:2].copy()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action, left_contact, right_contact, first_contact, air_time_snapshot, gravity, terminated):
        w = R.REWARD_WEIGHTS
        lin_vel_b = self._root_lin_vel_b()
        ang_vel_b = self._root_ang_vel_b()
        joint_pos = self.data.qpos[-self.nu:]
        phase_angle = 2.0 * np.pi * (
            (self.episode_t % GAIT_PERIOD) / GAIT_PERIOD
        )
######################################################################
        phase_left = phase_angle
        phase_right = phase_angle + np.pi
        left_foot_pos = self.data.xpos[self.left_foot_body_id].copy()
        right_foot_pos = self.data.xpos[self.right_foot_body_id].copy()
        leg_joint_pos = self.data.qpos[-self.nu:].copy()
        leg_joint_vel = self.data.qvel[-self.nu:].copy()
        actuator_force = self.data.actuator_force.copy()
        feet_z = min(left_foot_pos[2], right_foot_pos[2])
        base_height_rel = self.data.qpos[2] - feet_z
        #print(f"qpos_z={self.data.qpos[2]:.3f}, feet_z={feet_z:.3f}, base_height_rel={base_height_rel:.3f}")
#######################################################################
        terms = {
            "track_lin_vel_xy_exp": R.track_lin_vel_xy_exp(lin_vel_b[:2], self.cmd[:2], std=0.5),
            "track_ang_vel_z_exp": R.track_ang_vel_z_exp(ang_vel_b[2], self.cmd[2], std=0.5),
            "forward_progress": R.forward_progress(self.data.qpos[0:2], self.prev_base_pos_xy),
            "heading_drift": R.heading_drift_penalty(R.quat_to_yaw(self.data.qpos[3:7]), self.spawn_yaw),
            "lateral_path_deviation":R.lateral_path_deviation_penalty(
                self.data.qpos[0:2], self.spawn_xy, self.spawn_yaw
            ),
            "yaw_penalty":R.yaw_penalty(ang_vel_b[2],self.cmd[2]),
            "gait_phase_tracking":R.gait_phase_tracking_reward(
                phase_left,phase_right,float(left_contact),float(right_contact)
            ),
            "feet_air_time_reward": R.feet_air_time_reward(first_contact,air_time_snapshot,TARGET_FEET_AIR_TIME,self.cmd[:2]),   #############
            "symmetry":R.symmetry_penalty(leg_joint_pos,self.left_leg_pose_buffer,self.right_leg_pose_buffer),      ############
            "flat_orientation_l2": R.flat_orientation_l2(gravity),
            "base_height_l2": R.base_height_l2(base_height_rel, TARGET_HEIGHT),
            "pelvis_vel_tracking":R.pelvis_vel_tracking_penalty(lin_vel_b[:2],self.cmd[:2]),       #####
            "lateral_spread":R.lateral_spread_penalty(left_foot_pos,right_foot_pos),
            #########################################################################
            
            ##########################################
        
            "gait_phase_contact": R.gait_phase_contact_reward(self.episode_t, GAIT_PERIOD, left_contact, right_contact),

            "joint_pos_limits": R.joint_pos_limits(joint_pos, self.joint_lower, self.joint_upper),
            "joint_penalty": R.joint_penalty(leg_joint_pos, self.default_leg_joint_pos),
            "joint_vel": R.joint_vel_penalty(leg_joint_vel),
            "joint_acc": R.joint_acc_penalty(leg_joint_vel, self.prev_leg_joint_vel, self.dt_control),
            "torque": R.torque_penalty(actuator_force, leg_joint_vel),
            "action_rate_l2": R.action_rate_l2(action, self.prev_action),
            "action_smoothness2_l2": R.action_smoothness2_l2(action, self.prev_action, self.prev_prev_action),

            # "ang_vel_xy_l2": R.ang_vel_xy_l2(ang_vel_b),
            "alive_cost": R.alive_cost(),
            "is_terminated": float(terminated),
        }
        total = sum(w[k] * v for k, v in terms.items())
        return float(total), {"reward_terms": terms}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data
            )
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None