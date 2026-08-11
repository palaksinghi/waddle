"""
Single MuJoCo simulation containing N independent humanoids. One mj_step
call advances all N robots' physics together, using MuJoCo's own
vectorized C loop -- no subprocesses, no IPC.

Per-agent obs/reward/termination mirror CustomHumanoidEnv in humanoid_env.py.
Reset is per-agent: when a robot falls, only its own qpos/qvel slice is
reset in place; the rest of the sim (and other robots) keep running.

Rendering:
  - render_mode="human" opens a live, interactive MuJoCo viewer window
    (mujoco.viewer.launch_passive) and syncs it once per env.step() call
    (i.e. once per `frame_skip` physics substeps, not once per substep --
    syncing every substep would be far slower than useful).
  - render_mode="rgb_array" returns an offscreen-rendered frame per call,
    for recording videos; nothing calls this automatically, you must call
    env.render() yourself when in this mode.
  - Live "human" mode requires a real display (local monitor or X11
    forwarding). If the viewer window is closed by the user mid-run,
    _renderer_is_running() will report False so callers can fall back
    to headless training instead of crashing on a dead viewer handle.
"""

import os

import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from gymnasium.core import Env

from build_multi_xml import build_multi_humanoid_xml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class MultiHumanoidEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        n_agents,
        spacing=3.0,
        frame_skip=5,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        render_mode=None,
    ):
        self.n_agents = n_agents
        self.frame_skip = frame_skip
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.render_mode = render_mode

        xml_path = os.path.join(THIS_DIR, f"_multi_{n_agents}.xml")
        build_multi_humanoid_xml(n_agents, spacing=spacing, out_path=xml_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip

        # Per-agent index bookkeeping, resolved once from joint/actuator names.
        self._qpos_slices = []   # (start, end) into data.qpos, len = 7 + n_hinge
        self._qvel_slices = []   # (start, end) into data.qvel, len = 6 + n_hinge
        self._act_slices = []    # (start, end) into data.ctrl
        self._body_ids = []      # torso body id per agent, for cfrc_ext / xpos lookups

        for i in range(n_agents):
            root_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"root_{i}")
            qpos_start = self.model.jnt_qposadr[root_jid]
            qvel_start = self.model.jnt_dofadr[root_jid]

            # Each agent's joints were emitted contiguously by build_multi_xml
            # (freejoint + 21 hinges per robot, all appended together), so the
            # next agent's root joint id tells us where this agent's block ends.
            if i + 1 < n_agents:
                next_root_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"root_{i + 1}")
                qpos_end = self.model.jnt_qposadr[next_root_jid]
                qvel_end = self.model.jnt_dofadr[next_root_jid]
            else:
                qpos_end = self.model.nq
                qvel_end = self.model.nv

            self._qpos_slices.append((qpos_start, qpos_end))
            self._qvel_slices.append((qvel_start, qvel_end))

            act_start = i * (self.model.nu // n_agents)
            act_end = (i + 1) * (self.model.nu // n_agents)
            self._act_slices.append((act_start, act_end))

            self._body_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"torso_{i}"))

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        obs = self._get_agent_obs(0)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float64)
        self.single_action_space = spaces.Box(-1.0, 1.0, (self._act_slices[0][1] - self._act_slices[0][0],), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_agents,) + obs.shape, np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (n_agents,) + self.single_action_space.shape, np.float32)

        self._renderer = None
        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------------ #
    def _get_agent_obs(self, i):
        qs, qe = self._qpos_slices[i]
        vs, ve = self._qvel_slices[i]
        qpos = self.data.qpos[qs:qe].copy()[2:]  # drop x, y like the single-agent env
        qvel = self.data.qvel[vs:ve].copy()
        bid = self._body_ids[i]
        cinert = self.data.cinert[bid].copy()
        cvel = self.data.cvel[bid].copy()
        # qfrc_actuator / cfrc_ext are per-dof / per-body; slice to this agent's dofs/body
        qfrc_actuator = self.data.qfrc_actuator[vs:ve].copy()
        cfrc_ext = self.data.cfrc_ext[bid].copy()
        return np.concatenate([qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext])

    def _agent_z(self, i):
        qs, _ = self._qpos_slices[i]
        return self.data.qpos[qs + 2]

    def _agent_is_healthy(self, i):
        lo, hi = self._healthy_z_range
        return lo < self._agent_z(i) < hi

    def _agent_control_cost(self, action_i):
        return self._ctrl_cost_weight * np.sum(np.square(action_i))

    def _agent_contact_cost(self, i):
        bid = self._body_ids[i]
        cost = self._contact_cost_weight * np.sum(np.square(self.data.cfrc_ext[bid]))
        lo, hi = self._contact_cost_range
        return np.clip(cost, lo, hi)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self._reset_all_agents_noise()
        mujoco.mj_forward(self.model, self.data)
        obs = np.stack([self._get_agent_obs(i) for i in range(self.n_agents)])
        return obs, {}

    def _reset_all_agents_noise(self):
        lo, hi = -self._reset_noise_scale, self._reset_noise_scale
        for i in range(self.n_agents):
            qs, qe = self._qpos_slices[i]
            vs, ve = self._qvel_slices[i]
            self.data.qpos[qs:qe] += self._np_random.uniform(lo, hi, qe - qs)
            self.data.qvel[vs:ve] += self._np_random.uniform(lo, hi, ve - vs)

    def _reset_single_agent(self, i):
        """Reset only agent i's qpos/qvel slice in place, leaving everyone
        else's simulation state untouched (used on per-agent fall/reset)."""
        qs, qe = self._qpos_slices[i]
        vs, ve = self._qvel_slices[i]
        lo, hi = -self._reset_noise_scale, self._reset_noise_scale
        self.data.qpos[qs:qe] = self.init_qpos[qs:qe] + self._np_random.uniform(lo, hi, qe - qs)
        self.data.qvel[vs:ve] = self.init_qvel[vs:ve] + self._np_random.uniform(lo, hi, ve - vs)

    # ------------------------------------------------------------------ #
    def step(self, actions):
        """
        actions: array (n_agents, act_dim), each already in [-1, 1] (ctrlrange).
        Returns obs (n_agents, obs_dim), reward (n_agents,), terminated (n_agents,),
        truncated (n_agents,), info dict with per-agent breakdown.
        """
        actions = np.asarray(actions, dtype=np.float64)
        for i in range(self.n_agents):
            a_s, a_e = self._act_slices[i]
            self.data.ctrl[a_s:a_e] = np.clip(actions[i], -1.0, 1.0)

        xy_before = np.stack([self.data.qpos[qs:qs + 2].copy() for qs, _ in self._qpos_slices])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # ONE call steps every agent

        xy_after = np.stack([self.data.qpos[qs:qs + 2].copy() for qs, _ in self._qpos_slices])
        vel = (xy_after - xy_before) / self.dt

        obs = np.zeros((self.n_agents,) + self.single_observation_space.shape)
        reward = np.zeros(self.n_agents)
        terminated = np.zeros(self.n_agents, dtype=bool)
        forward_r = np.zeros(self.n_agents)
        ctrl_c = np.zeros(self.n_agents)
        contact_c = np.zeros(self.n_agents)

        for i in range(self.n_agents):
            forward_r[i] = self._forward_reward_weight * vel[i, 0]
            ctrl_c[i] = self._agent_control_cost(actions[i])
            contact_c[i] = self._agent_contact_cost(i)
            healthy = self._agent_is_healthy(i)
            reward[i] = forward_r[i] + (self._healthy_reward if healthy else 0.0) - ctrl_c[i] - contact_c[i]
            terminated[i] = not healthy
            obs[i] = self._get_agent_obs(i)

            # Per-agent auto-reset in place so one fallen robot doesn't end
            # the whole batch; the rest of the sim keeps running untouched.
            if terminated[i]:
                self._reset_single_agent(i)

        truncated = np.zeros(self.n_agents, dtype=bool)
        info = {
            "reward_forward": forward_r,
            "reward_ctrl": -ctrl_c,
            "reward_contact": -contact_c,
            "x_velocity": vel[:, 0],
            "y_velocity": vel[:, 1],
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                # launch_passive requires a real display; will raise if none
                # is available (e.g. fully headless server with no X server).
                self._renderer = mujoco.viewer.launch_passive(self.model, self.data)
            if self._renderer.is_running():
                self._renderer.sync()
            else:
                # user closed the viewer window -- stop trying to render
                # instead of throwing on every subsequent step() call.
                self.render_mode = None
            return None

        # rgb_array / offscreen mode: caller must invoke this explicitly.
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def _renderer_is_running(self):
        """Safe check for callers deciding whether to keep using this env
        in human-render mode (True if no viewer has been opened yet, since
        it will lazily open on the next render() call)."""
        if self.render_mode != "human":
            return False
        if self._renderer is None:
            return True
        return self._renderer.is_running()

    def close(self):
        if self._renderer is not None and hasattr(self._renderer, "close"):
            self._renderer.close()


def make_env(n_agents=11, render_mode=None):
    return MultiHumanoidEnv(n_agents=n_agents, render_mode="human")