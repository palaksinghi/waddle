# """Configuration for the Open Duck Mini v2 biped robot."""

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "open_duck_mini_v2")

# ---------------------------------------------------------------------------
# Base-link mass/inertia override.
#
# The source USD (open_duck_mini_v2_fixed.usd) authors an invalid inertia
# tensor {1,1,1} and a negative mass on .../Robot/base/base. Rather than hand
# -editing the USD, we set the correct values here in Python, applied once
# per environment right after the articulation is spawned into the scene
# (see BASE_MASS_FIX_EVENT below, wired into your EventCfg with mode="startup").
#
# Replace these with your real CAD/URDF <inertial> values if you have them;
# these are a solid-box approximation of the duck torso.
BASE_LINK_NAME = "base"
BASE_MASS_KG = 0.6
BASE_HALF_EXTENTS_M = (0.05, 0.04, 0.06)  # (x, y, z) half-extents of an equivalent box


def _box_inertia(mass: float, half_extents) -> torch.Tensor:
    dx, dy, dz = 2 * half_extents[0], 2 * half_extents[1], 2 * half_extents[2]
    ixx = (mass / 12.0) * (dy * dy + dz * dz)
    iyy = (mass / 12.0) * (dx * dx + dz * dz)
    izz = (mass / 12.0) * (dx * dx + dy * dy)
    return torch.tensor([ixx, iyy, izz], dtype=torch.float32)


def fix_base_mass_and_inertia(env, env_ids, asset_cfg):
    """Startup EventTerm: overwrite the base link's mass/inertia in the PhysX
    tensor view directly, replacing the invalid {1,1,1}/negative-mass values
    baked into the USD. Wire this into EventCfg as:

        fix_base_mass = EventTerm(
            func=fix_base_mass_and_inertia,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    Runs once at sim startup across all envs (env_ids covers every environment).
    """
    asset = env.scene[asset_cfg.name]
    body_idx = asset.body_names.index(BASE_LINK_NAME)

    num_envs = env.num_envs
    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()  # (num_envs, num_bodies, 9) flattened 3x3

    masses[:, body_idx] = BASE_MASS_KG

    diag = _box_inertia(BASE_MASS_KG, BASE_HALF_EXTENTS_M)
    inertia_mat = torch.zeros(9, dtype=torch.float32)
    inertia_mat[0] = diag[0]
    inertia_mat[4] = diag[1]
    inertia_mat[8] = diag[2]
    inertias[:, body_idx] = inertia_mat

    asset.root_physx_view.set_masses(masses, torch.arange(num_envs, device=masses.device))
    asset.root_physx_view.set_inertias(inertias, torch.arange(num_envs, device=inertias.device))

    print(
        f"[open_duck_mini_v2_cfg] Fixed '{BASE_LINK_NAME}' mass -> {BASE_MASS_KG} kg, "
        f"diagonal inertia -> {diag.tolist()} across {num_envs} envs"
    )


# ---------------------------------------------------------------------------
# Note on rootJoint_floor's disjointed transform warning:
#
# Isaac Lab writes the robot's root pose directly every reset via
# `write_root_pose_to_sim` (see EventCfg.reset_base in your env cfg, which
# calls mdp.reset_root_state_uniform) rather than driving it through
# rootJoint_floor's joint frame. That means the disjointed local
# pose on that joint does NOT affect training correctness here -- Isaac Lab
# overwrites the root link's world transform directly each reset, so the
# joint's authored (wrong) local frame is only used for the very first
# internal PhysX substep before the first reset call, which is what produces
# the log warning and a possible 1-frame "snap" you'd only notice if you're
# watching frame 0 in the viewer before any reset has run.
#
# If you still want it corrected at the source (recommended for cleanliness,
# and required if you ever spawn this asset in a context that does NOT reset
# root state via mdp.reset_root_state_uniform), the fix has to happen at the
# USD level since joint local-frame attributes aren't exposed through
# Isaac Lab's ArticulationCfg or PhysX tensor API the way per-body mass is.
# That's a one-time asset fix, not something to redo per training run.
# ---------------------------------------------------------------------------


OPEN_DUCK_MINI_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "open_duck_mini_v2_fixed.usd"),
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
        pos=(0.0, 0.0, 0.1934),
        joint_pos={
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.5,
            "left_knee": 1.0,
            "left_ankle": -0.5,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.5,
            "right_knee": 1.0,
            "right_ankle": -0.5,
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