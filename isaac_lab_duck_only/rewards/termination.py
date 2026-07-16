"""Termination terms for open_duck_mini_v2."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_angle_limit_termination(
    env: "ManagerBasedRLEnv",
    joint_names: list[str],
    max_deviation_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate an env instance once any tracked joint's angle deviates from its
    nominal/default position by more than `max_deviation_deg`, i.e. the joint has
    "fallen" past a safe range (e.g. knee/hip buckling, ankle over-rotation).

    Returns a boolean tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_names)

    q = asset.data.joint_pos[:, joint_ids]
    q_default = asset.data.default_joint_pos[:, joint_ids]

    max_dev_rad = torch.deg2rad(torch.tensor(max_deviation_deg, device=q.device))
    deviation = torch.abs(q - q_default)

    fell = torch.any(deviation > max_dev_rad, dim=1)
    return fell