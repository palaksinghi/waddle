"""Configuration for the Open Duck Mini v2 biped robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "open_duck_mini_v2")

OPEN_DUCK_MINI_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "open_duck_mini_v2_robot_only.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.0,
            "left_knee": 0.0,
            "left_ankle": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.0,
            "right_knee": 0.0,
            "right_ankle": 0.0,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "left_antenna": 0.0,
            "right_antenna": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_.*", "right_hip_.*", "left_knee", "right_knee", "left_ankle", "right_ankle"],
            effort_limit_sim=8.0,
            stiffness={
                "left_hip_yaw": 15.0,
                "right_hip_yaw": 15.0,
                "left_hip_roll": 15.0,
                "right_hip_roll": 15.0,
                "left_hip_pitch": 15.0,
                "right_hip_pitch": 15.0,
                "left_knee": 15.0,
                "right_knee": 15.0,
                "left_ankle": 10.0,
                "right_ankle": 10.0,
            },
            damping={
                "left_hip_yaw": 0.5,
                "right_hip_yaw": 0.5,
                "left_hip_roll": 0.5,
                "right_hip_roll": 0.5,
                "left_hip_pitch": 0.5,
                "right_hip_pitch": 0.5,
                "left_knee": 0.5,
                "right_knee": 0.5,
                "left_ankle": 0.3,
                "right_ankle": 0.3,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["neck_pitch", "head_.*", "left_antenna", "right_antenna"],
            effort_limit_sim=2.0,
            stiffness=5.0,
            damping=0.2,
        ),
    },
)

open_duck_mini_v2_cfg.py