
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
    (spawn_xy, spawn_yaw). Prevents circular/arcing paths -- position
    anchored, not velocity anchored."""
    dx = base_pos_xy[0] - spawn_xy[0]
    dy = base_pos_xy[1] - spawn_xy[1]
    lateral = -dx * np.sin(spawn_yaw) + dy * np.cos(spawn_yaw)
    return float(lateral ** 2)

def yaw_penalty(yaw_rate: float, cmd_yaw: float) -> float:
    err = (yaw_rate - cmd_yaw) ** 2
    return float(5.0 * np.tanh(err / 5.0))

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
    
def symmetry_penalty(leg_joint_pos: np.ndarray, delayed_leg_pose) -> float:
    """Penalize left/right leg joints not being mirrored appropriately
    given the half-cycle phase offset. Compares current leg pose (10-d:
    [left(5), right(5)]) to the pose recorded half a gait cycle ago,
    mirrored. Returns 0.0 until the history buffer is full."""
    if delayed_leg_pose is None:
        return 0.0
    mirror = np.array([1.0, -1.0, 1.0, 1.0, 1.0])  # yaw, roll, pitch, knee, ankle
    left_now = np.asarray(leg_joint_pos[0:5], dtype=np.float64)
    right_now = np.asarray(leg_joint_pos[5:10], dtype=np.float64)
    delayed = np.asarray(delayed_leg_pose, dtype=np.float64)
    target_right = delayed[0:5] * mirror
    target_left = delayed[5:10] * mirror
    err = np.sum((right_now - target_right) ** 2) + np.sum((left_now - target_left) ** 2)
    return float(err)

def flat_orientation_l2(projected_gravity: np.ndarray) -> float:
    return float(np.sum(projected_gravity[:2] ** 2))

def base_height_l2(height: float, target_height: float) -> float:
    return float((height - target_height) ** 2)


# def pelvis_vel_tracking_penalty(local_lin_vel_xy: np.ndarray, cmd_xy: np.ndarray) -> float:
#     """
#     p_v = ||v_p_xy - v_c||^2 / max(0.12, 0.5 * ||v_c||^2)
#
#     Speed-dependent tolerance: floor of 0.12 at low/zero commanded speed
#     (prevents exploding when standing still), scales with 0.5*||v_c||^2 at
#     higher commanded speed so the penalty isn't harsher than necessary.
#     """
#     err_sq = np.sum((local_lin_vel_xy - cmd_xy) ** 2)
#     denom = max(0.12, 0.5 * np.sum(cmd_xy ** 2))
#     return float(err_sq / denom)

def pelvis_vel_tracking_penalty(local_lin_vel_xy: np.ndarray, cmd_xy: np.ndarray) -> float:
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

############################################################################
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
    """Terminate if the robot's tilt exceeds tilt_limit (radians).

    projected_gravity[:2] has norm sin(tilt), so compare against
    sin(tilt_limit) rather than the raw radian value.
    """
    tilt = float(np.linalg.norm(projected_gravity[:2]))
    return bool(tilt > np.sin(tilt_limit))
#####################################################################################

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
    "track_lin_vel_xy_exp": 2.0,  #2.0
    "track_ang_vel_z_exp": 0.5,  #0.8
    "forward_progress": 8.0,

    # heading / straight-line
    "heading_drift": -1.0,    #-2.0
    "lateral_path_deviation": -2.0,    #-5.0
    "yaw_penalty":-1.0,   #-2.0

    # gait
    "gait_phase_tracking": 1.0,  #0.8
    "feet_air_time_reward": 2.0, #1.6
    "symmetry": -0.5,    #-0.9

    # base stability
    "flat_orientation_l2": -2.5,  #-1.0-->-2.5(21aug[2])
    "base_height_l2": -1.0,
#################################active
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
##################################
    "pelvis_vel_tracking": -1.0,   #-5.0
    "lateral_spread": -3.0, #-15
    "gait_phase_contact": 1.0,  #0.8
    "joint_pos_limits": -1.0,
    "joint_penalty": -0.001,   #-0.002
    "joint_vel": -0.0005,      #-0.001
    "joint_acc": -2.0e-7,  #7
    "torque": -0.0001, #.0002
    "action_rate_l2": -0.03,  #-0.05
    "action_smoothness2_l2": -0.015,   #-0.025

    # survival / termination
    "alive_cost": 1.0,
    "is_terminated": -25.0,
}
