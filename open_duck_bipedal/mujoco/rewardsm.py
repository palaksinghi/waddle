# import numpy as np
# ##
# #fix
# def phase_vector(t: float, period: float) -> np.ndarray:
#     phase = np.fmod(t, period) / period
#     ang = 2 * np.pi * phase
#     return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)

# #fix
# def track_lin_vel_xy_exp(lin_vel_xy: np.ndarray, cmd_xy: np.ndarray, std: float) -> float:
#     err = np.sum((cmd_xy - lin_vel_xy) ** 2)
#     return float(np.exp(-err / std**2))

# #fix
# def track_ang_vel_z_exp(ang_vel_z: float, cmd_z: float, std: float) -> float:
#     err = (cmd_z - ang_vel_z) ** 2
#     return float(np.exp(-err / std**2))


# def gait_phase_contact_reward(t, period, left_contact: bool, right_contact: bool) -> float:
#     ph = phase_vector(t, period)
#     left_should_contact = ph[0] >= 0.0
#     right_should_contact = not left_should_contact
#     left_match = float(left_contact == left_should_contact)
#     right_match = float(right_contact == right_should_contact)
#     return 0.5 * (left_match + right_match)

# def feet_air_time(last_air_time: np.ndarray, first_contact: np.ndarray, cmd_xy: np.ndarray, threshold=0.1) -> float:
#     reward = np.sum((last_air_time - threshold) * first_contact.astype(np.float32))
#     if np.linalg.norm(cmd_xy) <= 0.1:
#         reward = 0.0
#     else:
#         reward=0.5
#     return float(reward) ###if-else

# def flat_orientation_l2(projected_gravity: np.ndarray) -> float:
#     return float(np.sum(projected_gravity[:2] ** 2))

# def base_height_l2(height: float, target_height: float) -> float:
#     return float((height - target_height) ** 2)

# def joint_torques_l2(torques: np.ndarray) -> float:
#     return float(np.sum(torques ** 2))

# def joint_acc_l2(joint_acc: np.ndarray) -> float:
#     return float(np.sum(joint_acc ** 2))

# def action_rate_l2(action: np.ndarray, prev_action: np.ndarray) -> float:
#     return float(np.sum((action - prev_action) ** 2))

# def joint_pos_limits(joint_pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
#     out = -np.clip(joint_pos - lower, a_min=None, a_max=0.0)
#     out += np.clip(joint_pos - upper, a_min=0.0, a_max=None)
#     return float(np.sum(out))

# def undesired_contacts(contact_flags: np.ndarray) -> float:
#     return float(np.sum(contact_flags.astype(np.float32)))

# def bad_orientation(projected_gravity: np.ndarray, limit_angle: float) -> bool:
#     cos_tilt = -projected_gravity[2]
#     tilt_angle = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
#     return bool(tilt_angle > limit_angle)

# def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> float:
#     # q = [w, x, y, z]
#     dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
#     return float(2 * np.arccos(dot))

# def performance_reward(root_quat: np.ndarray, target_quat=(1.0, 0.0, 0.0, 0.0), omega=1.0, scale=0.1) -> float:
#     angle = quat_error_magnitude(root_quat, np.array(target_quat))
#     r_theta = np.sin(angle) ** 2 / scale
#     return float(omega * np.exp(-r_theta))
# # ----------------------------------------------------------------------
# # Added reward terms:
# # is_alive, symmetry, stable_head, stable_joint,
# # legal foot contact, collision penalty, action smoothness, torque energy
# # ----------------------------------------------------------------------
# def is_alive(terminated: bool) -> float:
#     """Small constant bonus every step the episode is still running.
#     Encourages the policy to survive longer instead of collapsing early."""
#     return float(0.0 if terminated else 1.0)

# def leg_yaw_stability_reward(
#     hip_yaw_pos: np.ndarray,      # left+right hip yaw joint angles
#     hip_yaw_default: np.ndarray,   # neutral/default yaw angles
#     scale: float = 0.05,
# ) -> float:
#     """Penalize hip yaw joints deviating from neutral, so legs don't
#     splay outward/inward during walking — keeps gait forward-directed."""
#     err = np.sum((hip_yaw_pos - hip_yaw_default) ** 2)
#     return float(np.exp(-err / scale))

# """abhikeliye"""
# # def gait_symmetry_reward(
# #     left_joint_pos: np.ndarray,
# #     right_joint_pos: np.ndarray,
# #     mirror_sign: np.ndarray | None = None,
# # ) -> float:
# #     """Penalize asymmetry between mirrored left/right joint angles during
# #     a gait cycle. left_joint_pos and right_joint_pos must be same-length
# #     arrays where index i on the left corresponds to the mirrored joint i
# #     on the right (e.g. [hip, knee, ankle]).

# #     mirror_sign: optional array of +1/-1 per joint, for joints whose sign
# #     convention flips between left and right sides (e.g. hip abduction).
# #     Defaults to all +1 (no sign flip).
# #     """
# #     if mirror_sign is None:
# #         mirror_sign = np.ones_like(left_joint_pos)
# #     diff = left_joint_pos - mirror_sign * right_joint_pos
# #     err = np.sum(diff ** 2)
# #     return float(np.exp(-err))

# # def stable_head_reward(
# #     head_joint_pos: np.ndarray,
# #     head_joint_default: np.ndarray,
# #     scale: float = 0.05,
# # ) -> float:
# #     """Penalize head/neck joint(s) deviating from their neutral position,
# #     so the head stays still instead of wobbling during locomotion."""
# #     err = np.sum((head_joint_pos - head_joint_default) ** 2)
# #     return float(np.exp(-err / scale))

# def stable_joint_reward(
#     joint_vel: np.ndarray,
#     scale: float = 1.0,
# ) -> float:
#     """Penalize high joint velocities / jitter, encouraging smooth,
#     stable joint motion rather than shaky high-frequency movement."""
#     err = np.sum(joint_vel ** 2)
#     return float(np.exp(-err / scale))

# # def legal_foot_contact_reward(
# #     left_contact: bool,
# #     right_contact: bool,
# #     left_force: float,
# #     right_force: float,
# #     max_force: float = 50.0,
# # ) -> float:
# #     """Reward feet making contact with the ground with reasonable force
# #     (not zero force "phantom" contact, not excessive slamming force).
# #     Returns 1.0 for each foot in legal contact, scaled down if force is
# #     outside the acceptable range."""
# #     def _leg_score(in_contact: bool, force: float) -> float:
# #         if not in_contact:
# #             return 0.0
# #         if force <= 0.0:
# #             return 0.0
# #         return float(np.clip(1.0 - abs(force - max_force / 2) / (max_force / 2), 0.0, 1.0))
# #     return 0.5 * (_leg_score(left_contact, left_force) + _leg_score(right_contact, right_force))

# def single_support_contact_reward(left_contact: bool, right_contact: bool) -> float:
#     """Reward exactly one foot on the ground while moving.
#     Reads the foot touch sensors directly: both feet planted (shuffling in
#     double support) or both feet airborne (hop/flight) score zero, so the
#     policy is pushed toward true alternating single-support stepping.
#     """
#     return float(left_contact != right_contact)  # XOR — True only if exactly one foot is down


# # def collision_penalty(contact_flags: np.ndarray) -> float:
# #     """Penalty for undesired body parts (torso, shins, elbows, etc.)
# #     touching the ground/obstacles. Same computation as undesired_contacts,
# #     named separately for clarity when used as its own reward term."""
# #     return float(np.sum(contact_flags.astype(np.float32)))


# # def action_smoothness_penalty(action: np.ndarray, prev_action: np.ndarray) -> float:
# #     """Penalize large jumps between consecutive actions, for smoother
# #     control output. Equivalent to action_rate_l2, named separately for
# #     reward-table clarity."""
# #     return float(np.sum((action - prev_action) ** 2))


# # def torque_energy_penalty(torques: np.ndarray, joint_vel: np.ndarray) -> float:
# #     """Penalize mechanical power / energy usage (torque * velocity),
# #     rather than just torque magnitude -- encourages energy-efficient gaits."""
# #     power = np.abs(torques * joint_vel)
# #     return float(np.sum(power))


# REWARD_WEIGHTS = {
#     "track_lin_vel_xy_exp": 6.0,  #1.5-->6.0
#     "track_ang_vel_z_exp": 0.75,
#     "gait_phase_contact": 0.5,  # WAS 0.3
#     "feet_air_time": 1.0,       # WAS 0.2
#     "flat_orientation_l2": -0.5,  # -1.0,#-2.0-->-0.5
#     "base_height_l2": -0.5,     # WAS -2.0 #-3.0-->-0.5
#     "joint_torques_l2": -2e-5,
#     "joint_acc_l2": -2.5e-7,
#     "action_rate_l2": -0.01,
#     "joint_pos_limits": -1.0,
#     "undesired_contacts": -1.0,
#     "is_terminated": -20.0,  #-100-->-20
#     "single_support_contact": 0.4,   # tune as needed
#     # newly added terms
#     "is_alive": 0.05,
#     "gait_symmetry": 0.3,
#     "stable_head": 0.2,
#     "stable_joint": 0.1,
#     "legal_foot_contact": 0.3,
#     "collision_penalty": -1.0,
#     "action_smoothness": -0.01,
#     "torque_energy": -1e-4,
# }
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