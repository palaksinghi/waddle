"""Extra observation terms not provided out-of-the-box by isaaclab.envs.mdp.

Used to build the 43-dim policy observation for the open_duck_mini_v2 walking
task:
    base_lin_vel (3) + base_ang_vel (3) + projected_gravity (3) + phase (2)
    + joint_pos (10) + joint_vel (10) + last_action (10) + feet_contact (2)
    = 43
Everything except `gait_phase_vector` and `feet_contact_bool` comes straight
from isaaclab.envs.mdp (base_lin_vel, base_ang_vel, projected_gravity,
joint_pos_rel, joint_vel_rel, last_action) - wired in duck_gait_env_cfg.py.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_vector(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Returns [sin(phase), cos(phase)], shape (N, 2)."""
    phase = env.gait_phase
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


def feet_contact_bool(
    env: "ManagerBasedRLEnv",
    contact_sensor_name: str = "feet_contact",
    left_foot_body_name: str = "left_foot",
    right_foot_body_name: str = "right_foot",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Returns [left_in_contact, right_in_contact] as 0/1 floats, shape (N, 2).

    Reads the same ContactSensor used by gait_phase_reward in reward.py, so the
    sensor only needs to be defined once in the scene cfg.
    """
    contact_sensor = env.scene.sensors[contact_sensor_name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, :, 2]  # (N, num_bodies) vertical force
    body_names = contact_sensor.body_names
    l_idx = body_names.index(left_foot_body_name)
    r_idx = body_names.index(right_foot_body_name)

    left_contact = (forces[:, l_idx] > force_threshold).float()
    right_contact = (forces[:, r_idx] > force_threshold).float()
    return torch.stack([left_contact, right_contact], dim=-1)