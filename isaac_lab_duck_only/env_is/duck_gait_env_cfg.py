"""ManagerBasedRLEnvCfg for open_duck_mini_v2 flat-ground gait."""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CommandTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp

from .obs_terms import gait_phase_vector
from robot.open_duck_mini_v2_cfg import OPEN_DUCK_MINI_V2_CFG
from rewards import reward, termination

NUM_JOINTS = 10
assert 11 + 2 * NUM_JOINTS == 31, "Observation term sizes no longer sum to 31 - update NUM_JOINTS or terms."

LEFT_LEG_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"]
RIGHT_LEG_JOINTS = ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
ALL_LEG_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS

BASE_HEIGHT_TARGET = 0.17 # m, adjust to your USD's standing height#27aug-->(2nd)


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class DuckSceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = OPEN_DUCK_MINI_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/(left_foot|right_foot)",
        history_length=3,
        track_air_time=True,
    )


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


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
# Events
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
# Rewards - matching MuJoCo-tuned weights
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.06, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.06, "asset_cfg": SceneEntityCfg("robot")},
    )
    forward_progress = RewTerm(func=reward.forward_progress, weight=8.0)

    heading_drift = RewTerm(func=reward.heading_drift_penalty, weight=-1.0)
    lateral_path_deviation = RewTerm(func=reward.lateral_path_deviation_penalty, weight=-4.0)
    yaw_penalty = RewTerm(func=reward.yaw_penalty, weight=-1.0, params={"command_name": "base_velocity"})

    gait_phase_tracking = RewTerm(
        func=reward.gait_phase_tracking_reward,
        weight=1.0,
        params={
            "left_foot_cfg": SceneEntityCfg("feet_contact", body_names="left_foot"),
            "right_foot_cfg": SceneEntityCfg("feet_contact", body_names="right_foot"),
            "sensor_cfg": SceneEntityCfg("feet_contact"),
        },
    )
    gait_phase_contact = RewTerm(
        func=reward.gait_phase_contact_reward,
        weight=1.0,
        params={
            "period": 0.7,
            "left_foot_cfg": SceneEntityCfg("feet_contact", body_names="left_foot"),
            "right_foot_cfg": SceneEntityCfg("feet_contact", body_names="right_foot"),
            "sensor_cfg": SceneEntityCfg("feet_contact"),
        },
    )
    feet_air_time = RewTerm(
        func=reward.feet_air_time,
        weight=2.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("feet_contact"), "threshold": 0.1},
    )
    symmetry = RewTerm(
        func=reward.symmetry_penalty,
        weight=-0.3,
        params={
            "left_joint_names": LEFT_LEG_JOINTS,
            "right_joint_names": RIGHT_LEG_JOINTS,
            "mirror_signs": [1.0] * len(LEFT_LEG_JOINTS),
        },
    )

    flat_orientation = RewTerm(func=reward.flat_orientation_l2, weight=-2.5)
    base_height = RewTerm(func=reward.base_height_l2, weight=-1.0, params={"target_height": BASE_HEIGHT_TARGET})
    lin_vel_z = RewTerm(func=reward.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=reward.ang_vel_xy_l2, weight=-0.05)
    pelvis_vel_tracking = RewTerm(
        func=reward.pelvis_vel_tracking_penalty, weight=-1.0, params={"command_name": "base_velocity"}
    )
    lateral_spread = RewTerm(
        func=reward.lateral_spread_penalty,
        weight=-3.0,
        params={
            "left_foot_cfg": SceneEntityCfg("robot", body_names="left_foot"),
            "right_foot_cfg": SceneEntityCfg("robot", body_names="right_foot"),
        },
    )

    joint_pos_limits = RewTerm(
        func=reward.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)}
    )
    joint_penalty = RewTerm(
        func=reward.joint_deviation_from_default, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)}
    )
    joint_vel = RewTerm(
        func=reward.joint_vel_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)}
    )
    joint_acc = RewTerm(
        func=reward.joint_acc_l2, weight=-2.0e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)}
    )
    torque = RewTerm(
        func=reward.torque_penalty, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_LEG_JOINTS)}
    )
    action_rate = RewTerm(func=reward.action_rate_l2, weight=-0.03)
    action_smoothness = RewTerm(func=reward.action_smoothness2_l2, weight=-0.015)

    alive_cost = RewTerm(func=reward.alive_reward, weight=1.0)
    is_terminated = RewTerm(func=reward.is_terminated, weight=-25.0)


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
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
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    gait_cycle_period_s: float = 0.7

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)