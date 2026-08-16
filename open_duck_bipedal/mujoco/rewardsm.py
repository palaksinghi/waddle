import numpy as np
#1
def phase_vector(t: float, period: float) -> np.ndarray:
    phase = np.fmod(t, period) / period
    ang = 2 * np.pi * phase
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)
#2
def quat_to_yaw(quat: np.ndarray) -> float:
    """quat = [w, x, y, z] -> yaw angle (rotation about world z-axis)."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
#3
def bad_orientation(projected_gravity: np.ndarray, limit_angle: float) -> bool:
    cos_tilt = -projected_gravity[2]
    tilt_angle = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
    return bool(tilt_angle > limit_angle)
# Forward/sideways velocity tracking
#4
def track_lin_vel_xy_exp(lin_vel_xy: np.ndarray, cmd_xy: np.ndarray, std: float) -> float:
    err = np.sum((cmd_xy - lin_vel_xy) ** 2)
    return float(np.exp(-err / std**2))
# Turning (yaw rate) tracking
#5
def track_ang_vel_z_exp(ang_vel_z: float, cmd_z: float, std: float) -> float:
    err = (cmd_z - ang_vel_z) ** 2
    return float(np.exp(-err / std**2))
# Heading lock — keeps the robot walking in a straight line
#6
def heading_penalty(current_yaw: float, initial_yaw: float, cmd_wz: float, turn_threshold: float = 0.05) -> float:
    """Penalize yaw drifting away from the heading recorded at episode
    start, whenever no turning is commanded. This is what stops the
    robot curving/circling — tracking_ang_vel alone only fights the
    instantaneous rate, small per-step errors still accumulate into a
    large heading drift over an episode."""
    if abs(cmd_wz) > turn_threshold:
        return 0.0
    err = (current_yaw - initial_yaw) ** 2
    return float(err)
# Gait phase / foot contact matching
#7
def gait_phase_contact_reward(t, period, left_contact: bool, right_contact: bool) -> float:
    ph = phase_vector(t, period)
    left_should_contact = ph[0] >= 0.0
    right_should_contact = not left_should_contact
    left_match = float(left_contact == left_should_contact)
    right_match = float(right_contact == right_should_contact)
    return 0.5 * (left_match + right_match)
#  Feet air time (stepping reward)
#8
def feet_air_time(last_air_time: np.ndarray, first_contact: np.ndarray, cmd_xy: np.ndarray, threshold=0.1) -> float:
    reward = np.sum((last_air_time - threshold) * first_contact.astype(np.float32))
    if np.linalg.norm(cmd_xy) <= 0.1:
        reward = 0.0
    return float(reward)
#  Flat orientation (torso tilt) penalty
#9
def flat_orientation_l2(projected_gravity: np.ndarray) -> float:
    return float(np.sum(projected_gravity[:2] ** 2))
# Base height penalty
#10
def base_height_l2(height: float, target_height: float) -> float:
    return float((height - target_height) ** 2)
#  Action rate (jerkiness) penalty
#11
def action_rate_l2(action: np.ndarray, prev_action: np.ndarray) -> float:
    return float(np.sum((action - prev_action) ** 2))
# Joint position limits penalty
#11
def joint_pos_limits(joint_pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    out = -np.clip(joint_pos - lower, a_min=None, a_max=0.0)
    out += np.clip(joint_pos - upper, a_min=0.0, a_max=None)
    return float(np.sum(out))

# 10. Termination penalty — handled directly as float(terminated) in envm.py
# (no separate function needed — envm.py already does: "is_terminated": float(terminated))
# Weights — tuned so walking dominates over standing-still
REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 6.0, #5-->6
    "track_ang_vel_z_exp": 1.0,
    "heading": -2.5,
    "gait_phase_contact": 0.3,
    "feet_air_time": 1.2,
    "flat_orientation_l2": -1.0,
    "base_height_l2": -2.0,
    "action_rate_l2": -0.01,
    "joint_pos_limits": -1.0,
    "is_terminated": -20.0,
}