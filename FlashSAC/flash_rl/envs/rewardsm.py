import numpy as np


def track_lin_vel_xy_exp(lin_vel_xy: np.ndarray, cmd_xy: np.ndarray, std: float) -> float:
    err = np.sum((cmd_xy - lin_vel_xy) ** 2)
    return float(np.exp(-err / std**2))


def track_ang_vel_z_exp(ang_vel_z: float, cmd_z: float, std: float) -> float:
    err = (cmd_z - ang_vel_z) ** 2
    return float(np.exp(-err / std**2))


def forward_progress(pos_xy: np.ndarray, prev_pos_xy: np.ndarray) -> float:
    """Reward net forward displacement (world +x) this step."""
    return float(pos_xy[0] - prev_pos_xy[0])


def heading_drift_penalty(base_yaw: float, spawn_yaw: float) -> float:
    """Yaw deviation from the heading at spawn. Anchored to a fixed
    reference (spawn_yaw), not recomputed each step."""
    err = _wrap_to_pi(base_yaw - spawn_yaw)
    return float(err ** 2)


def lateral_path_deviation_penalty(base_pos_xy: np.ndarray, spawn_xy: np.ndarray, spawn_yaw: float) -> float:
    """Perpendicular distance from the straight line defined by
    (spawn_xy, spawn_yaw). Prevents cicular/arcing paths -- position
    anchored, not velocity anchored."""
    dx = base_pos_xy[0] - spawn_xy[0]
    dy = base_pos_xy[1] - spawn_xy[1]
    lateral = -dx * np.sin(spawn_yaw) + dy * np.cos(spawn_yaw)
    return float(lateral ** 2)


def yaw_penalty(yaw_rate: float, cmd_yaw: float) -> float:
    err = (yaw_rate - cmd_yaw) ** 2
    return float(np.clip(err, 0.0, 5.0))


def gait_phase_tracking_reward(phase_left: float, phase_right: float,
                                left_contact: float, right_contact: float) -> float:
    """Alternate formulation: desired stance derived from sin(phase) sign
    per-leg (legs pi apart), matched against actual contact state."""
    desired_left_stance = 1.0 if np.sin(phase_left) > 0 else 0.0
    desired_right_stance = 1.0 if np.sin(phase_right) > 0 else 0.0
    left_match = 1.0 - abs(desired_left_stance - left_contact)
    right_match = 1.0 - abs(desired_right_stance - right_contact)
    return float(0.5 * (left_match + right_match))


def feet_air_time_reward(foot_touchdown_event, foot_air_time, target_feet_air_time: float,
                          cmd_xy: np.ndarray) -> float:
    """Pays out on touchdown events, capped at target air time; zero if no
    forward/lateral command is active."""
    if np.linalg.norm(cmd_xy) < 0.05:
        return 0.0
    r = 0.0
    for i in range(len(foot_touchdown_event)):
        if foot_touchdown_event[i]:
            r += min(foot_air_time[i], target_feet_air_time)
    return float(r)


def symmetry_penalty(leg_joint_pos: np.ndarray, left_leg_pose_buffer, right_leg_pose_buffer) -> float:
    """Penalize left/right leg joints not being mirrored appropriately
    given the half-cycle phase offset. Compares current left-leg pose to
    the right-leg pose recorded half a gait cycle ago (and vice versa).

    NOTE: this only produces a nonzero value once the env has actually
    populated left_leg_pose_buffer / right_leg_pose_buffer (e.g. every
    half gait cycle in step()). If the env never updates those buffers,
    this always returns 0.0 -- see the "symmetry" bug flagged in the env
    file review.
    """
    if left_leg_pose_buffer is None or right_leg_pose_buffer is None:
        return 0.0
    mirror = np.array([1.0, -1.0, 1.0, 1.0, 1.0])  # yaw, roll, pitch, knee, ankle
    left_now = leg_joint_pos[0:5]
    right_now = leg_joint_pos[5:10]
    target_right = left_leg_pose_buffer * mirror
    target_left = right_leg_pose_buffer * mirror
    err = np.sum((right_now - target_right) ** 2) + np.sum((left_now - target_left) ** 2)
    return float(err)


def flat_orientation_l2(projected_gravity: np.ndarray) -> float:
    return float(np.sum(projected_gravity[:2] ** 2))


def base_height_l2(height: float, target_height: float) -> float:
    return float((height - target_height) ** 2)


def pelvis_vel_tracking_penalty(local_lin_vel_xy: np.ndarray, cmd_xy: np.ndarray) -> float:
    """
    p_v = ||v_p_xy - v_c||^2 / max(0.12, 0.5 * ||v_c||^2)

    Speed-dependent tolerance: floor of 0.12 at low/zero commanded speed
    (prevents exploding when standing still), scales with 0.5*||v_c||^2 at
    higher commanded speed so the penalty isn't harsher than necessary.
    Clipped to avoid extreme outlier penalties dominating the total.
    """
    err_sq = np.sum((local_lin_vel_xy - cmd_xy) ** 2)
    denom = max(0.12, 0.5 * np.sum(cmd_xy ** 2))
    val = err_sq / denom
    return float(np.clip(val, 0.0, 5.0))


def lateral_spread_penalty(left_foot_pos: np.ndarray, right_foot_pos: np.ndarray, max_spread: float = 0.25) -> float:
    """Penalize the lateral (y-axis) distance between the feet exceeding
    max_spread."""
    lateral_distance = abs(left_foot_pos[1] - right_foot_pos[1])
    over = max(0.0, lateral_distance - max_spread)
    return float(over)


def gait_phase_contact_reward(t, period, left_contact: bool, right_contact: bool) -> float:
    ph = phase_vector(t, period)
    left_should_contact = ph[0] >= 0.0
    right_should_contact = not left_should_contact
    left_match = float(left_contact == left_should_contact)
    right_match = float(right_contact == right_should_contact)
    return 0.5 * (left_match + right_match)


def joint_pos_limits(joint_pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    out = -np.clip(joint_pos - lower, a_min=None, a_max=0.0)
    out += np.clip(joint_pos - upper, a_min=0.0, a_max=None)
    return float(np.sum(out))


def joint_penalty(leg_joint_pos: np.ndarray, default_leg_joint_pos: np.ndarray) -> float:
    """Sum of squared deviation of leg joints from the default/home pose."""
    return float(np.sum((leg_joint_pos - default_leg_joint_pos) ** 2))


def joint_vel_penalty(leg_joint_vel: np.ndarray) -> float:
    return float(np.sum(leg_joint_vel ** 2))


def joint_acc_penalty(leg_joint_vel: np.ndarray, prev_leg_joint_vel: np.ndarray, dt_control: float) -> float:
    acc = (leg_joint_vel - prev_leg_joint_vel) / dt_control
    return float(np.sum(acc ** 2))


def torque_penalty(actuator_force: np.ndarray, qvel_actuated: np.ndarray) -> float:
    """Approximate mechanical power as |torque * joint_vel|, summed."""
    power = np.abs(actuator_force * qvel_actuated)
    return float(np.sum(power))


def phase_vector(t: float, period: float) -> np.ndarray:
    phase = np.fmod(t, period) / period
    ang = 2 * np.pi * phase
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def bad_orientation(projected_gravity: np.ndarray, tilt_limit: float) -> bool:
    """Terminate if the robot's projected gravity indicates excessive tilt."""
    tilt = np.linalg.norm(projected_gravity[:2])
    return bool(tilt > tilt_limit)


def ang_vel_xy_l2(ang_vel_b: np.ndarray) -> float:
    return float(np.sum(ang_vel_b[:2] ** 2))


def lin_vel_z_l2(local_lin_vel_z: float) -> float:
    return float(local_lin_vel_z ** 2)


def action_rate_l2(action: np.ndarray, prev_action: np.ndarray) -> float:
    return float(np.sum((action - prev_action) ** 2))


def action_smoothness2_l2(action: np.ndarray, prev_action: np.ndarray, prev_prev_action: np.ndarray) -> float:
    """Second-order smoothness: penalizes acceleration in action space."""
    return float(np.sum((action - 2 * prev_action + prev_prev_action) ** 2))


def alive_cost() -> float:
    return 1.0


REWARD_WEIGHTS = {
    # tracking
    "track_lin_vel_xy_exp": 2.0,
    "track_ang_vel_z_exp": 0.5,
    "forward_progress": 8.0,

    # heading / straight-line
    "heading_drift": -1.0,
    "lateral_path_deviation": -4.0,
    "yaw_penalty": -1.0,

    # gait
    "gait_phase_tracking": 1.0,
    "feet_air_time_reward": 2.0,
    "symmetry": -0.3,

    # base stability
    "flat_orientation_l2": -2.5,
    "base_height_l2": -1.0,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,

    "pelvis_vel_tracking": -1.0,
    "lateral_spread": -3.0,
    "gait_phase_contact": 1.0,
    "joint_pos_limits": -1.0,
    "joint_penalty": -0.001,
    "joint_vel": -0.0005,
    "joint_acc": -2.0e-7,
    "torque": -0.0001,
    "action_rate_l2": -0.03,
    "action_smoothness2_l2": -0.015,

    # survival / termination
    "alive_cost": 1.0,
    "is_terminated": -25.0,
}

# NOTE: `lin_vel_z_l2` and `ang_vel_xy_l2` weights above are only meaningful
# if the env's _compute_reward() actually includes those terms in its
# `terms` dict (they were missing/commented out in the original env file --
# see open_duck_mini_v2.py where they've been wired back in). If you decide
# NOT to use them, remove the two lines above instead of leaving dead weights.

# The previous version of this file also had a standalone `compute_reward(e)`
# function at the bottom that referenced an `e` object with attributes
# (e.g. e.local_lin_vel, e.commands, e.reward_cfg) that don't exist anywhere
# in OpenDuckBipedalEnv -- it was never called and has been removed here.
# The env computes reward itself via `_compute_reward()`, calling the
# individual functions above directly.r
