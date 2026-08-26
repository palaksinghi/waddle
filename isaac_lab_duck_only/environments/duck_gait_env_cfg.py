"""ManagerBasedRLEnvCfg for open_duck_mini_v2 flat-ground gait.

Observation layout (43-dim total) — EDIT `NUM_JOINTS` if your USD's actuated
joint count differs; the assertion below will fail loudly if the math doesn't
add up so you catch it before launching training:

    base_ang_vel        3
    base_lin_vel        3
    projected_gravity   3
    gait_phase (sin,cos)2
    joint_pos            N
    joint_vel            N
  -----------------------------
    total = 11 + 2*N  ==  43   ->  N = 16
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp

from .obs_terms import gait_phase_vector
from robot.open_duck_mini_v2_cfg import OPEN_DUCK_MINI_V2_CFG  # your existing ArticulationCfg
from rewards import reward, termination

NUM_JOINTS = 16  # <-- set to your USD's actuated joint count
assert 11 + 2 * NUM_JOINTS == 43, "Observation term sizes no longer sum to 43 - update NUM_JOINTS or terms."

LEFT_LEG_JOINTS = ["left_hip_.*", "left_knee.*", "left_ankle.*"]
RIGHT_LEG_JOINTS = ["right_hip_.*", "right_knee.*", "right_ankle.*"]
ALL_LEG_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS

BASE_HEIGHT_TARGET = 0.20  # m, nominal standing height of open_duck_mini_v2 - adjust to your USD


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class DuckSceneCfg(InteractiveSceneCfg):
    terrain = sim_utils.GroundPlaneCfg()
    robot = OPEN_DUCK_MINI_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    feet_contact = None  # set to your ContactSensorCfg on the feet bodies in your existing scene file


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ALL_LEG_JOINTS, scale=1.0, use_default_offset=True
    )


# -----------------------------------------------------------------------------
# Observations (43-dim)
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        phase = ObsTerm(func=gait_phase_vector)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Events (resets only - no domain randomization / curriculum per your request)
# -----------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )


# -----------------------------------------------------------------------------
# Rewards - all weights live here, nowhere else
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    forward_progress = RewTerm(func=reward.forward_progress, weight=2.0)
    flat_orientation = RewTerm(func=reward.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=reward.base_height_l2, weight=-1.0, params={"target_height": BASE_HEIGHT_TARGET})
    lin_vel_z = RewTerm(func=reward.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=reward.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=reward.joint_torques_l2, weight=-1e-5)
    joint_acc = RewTerm(func=reward.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=reward.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=reward.action_smoothness2_l2, weight=-0.01)
    symmetry = RewTerm(
        func=reward.symmetry_penalty,
        weight=-0.5,
        params={
            "left_joint_names": LEFT_LEG_JOINTS,
            "right_joint_names": RIGHT_LEG_JOINTS,
            "mirror_signs": [1.0] * len(LEFT_LEG_JOINTS),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_fell = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": BASE_HEIGHT_TARGET * 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )


# -----------------------------------------------------------------------------
# Env cfg
# -----------------------------------------------------------------------------
@configclass
class DuckGaitEnvCfg(ManagerBasedRLEnvCfg):
    scene: DuckSceneCfg = DuckSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    gait_cycle_period_s: float = 0.7  # full left+right stance cycle duration

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation