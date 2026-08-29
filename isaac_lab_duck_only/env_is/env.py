#rel-->relative from current to default
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg    #observation,action,reward,termiation
# from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import os
from pathlib import Path
from . import mdp
from .robot_cfg import OPEN_DUCK_CFG

_DEFAULT_USD_PATH=Path(__file__).resolve().parents[2]/"robot"/"openduck.usd"
OPENDUCK_USD_PATH=os.environ.get("OPENDUCK_USD_PATH", str(_DEFAULT_USD_PATH))
# Scene config
@configclass
class OpenDuckSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",   
        )

    robot: ArticulationCfg = OPEN_DUCK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenDuck/.*",
        spawn=sim_utils.UsdFileCfg(usd_path=OPENDUCK_USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0,0.0,0.45)))  #articulationconfig

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/OpenDuck",  #in isaaclab we need to write like this
        history_length=3,
        track_air_time=True
    )

    sky_light = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=None,
        color=(0.9, 0.9, 0.9)
    )

# MDP 
@configclass
class CommandsCfg:   #what the robot should do is-->the command we have to give
    base_velocity = mdp.UniformVelocityCommandCfg(   #velocity command
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),    #in every 4-6 seconds the command can change 
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
#actioncfg
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5,
    )
#observationscfg
@configclass
class ObservationsCfg:
    #policyconfig
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

#EVENTCONFIG
@configclass
class EventCfg:   #reset,push
    ##Domain randomization+episodereset

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0,0)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.0, 0.0),"velocity_range": (0.0, 0.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )
    reset_reward_buffers = EventTerm(
        func=mdp.reset_reward_buffers,
        mode="reset",
    )

#rewardconfig
@configclass
class RewardsCfg:
    # task: velocity tracking 
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0, 
        params={"std": 0.06, "command_name": "base_velocity"}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"std": 0.06, "command_name": "base_velocity"}
    )
    forward_progress = RewTerm(func=mdp.forward_progress, weight=8.0)

    heading_drift = RewTerm(func=mdp.heading_drift_penalty, weight=-1.0)
    lateral_path_deviation = RewTerm(func=mdp.lateral_path_deviation_penalty, weight=-4.0)
    yaw_penalty = RewTerm(
        func=mdp.yaw_penalty, weight=-1.0, params={"command_name": "base_velocity"}
    )

    gait_contact = RewTerm(
        func=mdp.gait_phase_contact_reward,
        weight=0.5,
        params={
            "period": 0.6,  
            "left_foot_cfg": SceneEntityCfg("contact_forces", body_names="left_foot"),
            "right_foot_cfg": SceneEntityCfg("contact_forces", body_names="right_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )
    symmetry = RewTerm(
        func=mdp.symmetry_penalty,
        weight=-0.3,
        params={
            "left_joint_names": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"],
            "right_joint_names": ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"],
            "mirror_signs": [1.0, -1.0, 1.0, 1.0, 1.0],
        },
    )
    
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)  #penalizes the robot from being tilt hence negative weights
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.2})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pelvis_vel_tracking = RewTerm(
        func=mdp.pelvis_vel_tracking_penalty,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    lateral_spread = RewTerm(
        func=mdp.lateral_spread_penalty,
        weight=-3.0,
        params={
            "left_foot_cfg": SceneEntityCfg("robot", body_names="left_foot"),
            "right_foot_cfg": SceneEntityCfg("robot", body_names="right_foot"),
            "max_spread": 0.25,
        },
    )

    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)  
    joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    action_smoothness2 = RewTerm(func=mdp.action_smoothness2_l2, weight=-0.015)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*(hip|knee|base).*"]),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-25.0)

#Terminationconfig
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_fell = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 0.7}  
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
# Environment configuration
@configclass
class OpenDuckFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: OpenDuckSceneCfg = OpenDuckSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.disable_contact_processing = True

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class OpenDuckFlatEnvCfg_PLAY(OpenDuckFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)