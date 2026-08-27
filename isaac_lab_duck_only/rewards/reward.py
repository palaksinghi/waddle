from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking the commanded xy linear velocity
    (expressed in the base/local frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_err = torch.sum(
        torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_err / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking the commanded yaw rate."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_err = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_err / std**2)


def forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Net forward displacement (world +x) since the previous step.

    Requires the previous root position to be cached on the env; see
    `_update_prev_pos_xy` below, called from EventCfg on an interval,
    or track it yourself in a custom env subclass.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    prev_pos_xy = getattr(env, "_prev_root_pos_xy", None)
    cur_pos_xy = asset.data.root_pos_w[:, :2]
    if prev_pos_xy is None:
        env._prev_root_pos_xy = cur_pos_xy.clone()
        return torch.zeros(env.num_envs, device=env.device)
    progress = cur_pos_xy[:, 0] - prev_pos_xy[:, 0]
    env._prev_root_pos_xy = cur_pos_xy.clone()
    return progress



def heading_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared yaw deviation from the heading recorded at spawn."""
    asset: Articulation = env.scene[asset_cfg.name]
    spawn_yaw = _get_spawn_yaw(env, asset)
    cur_yaw = _quat_to_yaw(asset.data.root_quat_w)
    err = wrap_to_pi(cur_yaw - spawn_yaw)
    return torch.square(err)


def lateral_path_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared perpendicular distance from the straight line defined by
    the spawn position and spawn heading."""
    asset: Articulation = env.scene[asset_cfg.name]
    spawn_xy = _get_spawn_xy(env, asset)
    spawn_yaw = _get_spawn_yaw(env, asset)
    cur_xy = asset.data.root_pos_w[:, :2]
    dx = cur_xy[:, 0] - spawn_xy[:, 0]
    dy = cur_xy[:, 1] - spawn_xy[:, 1]
    lateral = -dx * torch.sin(spawn_yaw) + dy * torch.cos(spawn_yaw)
    return torch.square(lateral)


def yaw_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_err: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Clipped squared error between actual and commanded yaw rate."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.square(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2])
    return torch.clamp(err, 0.0, max_err)


def gait_phase_contact_reward(
    env: ManagerBasedRLEnv,
    period: float,
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching actual foot contact state to the desired gait
    phase (legs pi apart, alternating stance)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w_history  # (N, hist, B, 3)

    left_forces = torch.norm(
        net_forces[:, :, left_foot_cfg.body_ids, :], dim=-1
    ).max(dim=1)[0].squeeze(-1)
    right_forces = torch.norm(
        net_forces[:, :, right_foot_cfg.body_ids, :], dim=-1
    ).max(dim=1)[0].squeeze(-1)
    left_contact = left_forces > threshold
    right_contact = right_forces > threshold

    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.fmod(t, period) / period
    ang = 2.0 * torch.pi * phase
    left_should_contact = torch.sin(ang) >= 0.0
    right_should_contact = ~left_should_contact

    left_match = (left_contact == left_should_contact).float()
    right_match = (right_contact == right_should_contact).float()
    return 0.5 * (left_match + right_match)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Reward long swing phases (air time) on touchdown, capped at
    `threshold`, gated off when no xy velocity is commanded."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    cmd = env.command_manager.get_command(command_name)
    reward *= torch.norm(cmd[:, :2], dim=1) > 0.05
    return reward


def symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_joint_names: list[str],
    right_joint_names: list[str],
    mirror_signs: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left/right leg joints deviating from a mirrored pose.

    Simplified vs. a half-cycle-delayed buffer version: compares the
    current left pose to the mirrored current right pose. Swap in your
    own phase-delayed buffers on the env if you need the delayed variant.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    left_ids, _ = asset.find_joints(left_joint_names)
    right_ids, _ = asset.find_joints(right_joint_names)
    mirror = torch.tensor(mirror_signs, device=env.device)

    left_pos = asset.data.joint_pos[:, left_ids]
    right_pos = asset.data.joint_pos[:, right_ids]
    err = torch.sum(torch.square(left_pos - right_pos * mirror), dim=1)
    return err


def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def pelvis_vel_tracking_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    floor: float = 0.12,
    max_err: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Speed-scaled tracking penalty: err / max(floor, 0.5*||cmd||^2),
    clipped so it can't explode."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err_sq = torch.sum(
        torch.square(asset.data.root_lin_vel_b[:, :2] - cmd[:, :2]), dim=1
    )
    denom = torch.clamp(0.5 * torch.sum(torch.square(cmd[:, :2]), dim=1), min=floor)
    return torch.clamp(err_sq / denom, 0.0, max_err)


def lateral_spread_penalty(
    env: ManagerBasedRLEnv,
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    max_spread: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    left_y = asset.data.body_pos_w[:, left_foot_cfg.body_ids[0], 1]
    right_y = asset.data.body_pos_w[:, right_foot_cfg.body_ids[0], 1]
    over = torch.clamp(torch.abs(left_y - right_y) - max_spread, min=0.0)
    return over


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    out_of_limits = -(pos - lower).clip(max=0.0)
    out_of_limits += (pos - upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def action_smoothness2_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order smoothness penalty. Requires caching the
    previous-previous action on the env (done here automatically)."""
    prev_prev = getattr(env, "_prev_prev_action", None)
    cur = env.action_manager.action
    prev = env.action_manager.prev_action
    if prev_prev is None:
        env._prev_prev_action = prev.clone()
        return torch.zeros(env.num_envs, device=env.device)
    penalty = torch.sum(torch.square(cur - 2 * prev + prev_prev), dim=1)
    env._prev_prev_action = prev.clone()
    return penalty


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on envs that terminated this step for a non-timeout reason."""
    return (env.termination_manager.terminated).float()


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact.float(), dim=1)

def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return tilt > limit_angle


def illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.any(
        torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > threshold, dim=1
    )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _get_spawn_yaw(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    """Caches the yaw at the last reset per-env. Update this buffer from
    an EventTerm with mode='reset' if you want it recomputed on reset;
    here it lazily initializes on first access."""
    if not hasattr(env, "_spawn_yaw"):
        env._spawn_yaw = _quat_to_yaw(asset.data.root_quat_w).clone()
    return env._spawn_yaw


def _get_spawn_xy(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    if not hasattr(env, "_spawn_xy"):
        env._spawn_xy = asset.data.root_pos_w[:, :2].clone()
    return env._spawn_xy

# --------------------------------------------------------------------------
# Added to match MuJoCo-tuned reward set
# --------------------------------------------------------------------------

def joint_deviation_from_default(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of squared deviation of joints from their default (home) pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(pos - default), dim=1)


def joint_vel_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def torque_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Approximate mechanical power as |torque * joint_vel|, summed."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * vel), dim=1)


def alive_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def gait_phase_tracking_reward(
    env: "ManagerBasedRLEnv",
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Alternate gait-phase formulation: desired stance derived from
    sin(phase) sign per-leg (legs pi apart), matched against actual
    contact state. Uses env.gait_phase (set on the env each step)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w_history

    left_forces = torch.norm(
        net_forces[:, :, left_foot_cfg.body_ids, :], dim=-1
    ).max(dim=1)[0].squeeze(-1)
    right_forces = torch.norm(
        net_forces[:, :, right_foot_cfg.body_ids, :], dim=-1
    ).max(dim=1)[0].squeeze(-1)
    left_contact = (left_forces > threshold).float()
    right_contact = (right_forces > threshold).float()

    phase = getattr(env, "gait_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, device=env.device)

    desired_left_stance = (torch.sin(phase) > 0).float()
    desired_right_stance = 1.0 - desired_left_stance

    left_match = 1.0 - torch.abs(desired_left_stance - left_contact)
    right_match = 1.0 - torch.abs(desired_right_stance - right_contact)
    return 0.5 * (left_match + right_match)