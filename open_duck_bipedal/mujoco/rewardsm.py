
import numpy as np

#  Forward/sideways velocity tracking (exp kernel)
def track_lin_vel_xy_exp(lin_vel_xy: np.ndarray, cmd_xy: np.ndarray, std: float) -> float:
    err = np.sum((cmd_xy - lin_vel_xy) ** 2)
    return float(np.exp(-err / std**2))

#  Turning (yaw rate) tracking (exp kernel)
def track_ang_vel_z_exp(ang_vel_z: float, cmd_z: float, std: float) -> float:
    err = (cmd_z - ang_vel_z) ** 2
    return float(np.exp(-err / std**2))

# Forward progress
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
    """Squared error between actual and commanded yaw rate. With no turn
    commanded this punishes any yawing at all, so the robot walks straight
    instead of circling. Turning exactly as commanded costs nothing."""
    return float((yaw_rate - cmd_yaw) ** 2)

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
    # Symmetry
def symmetry_penalty(leg_joint_pos: np.ndarray, left_leg_pose_buffer, right_leg_pose_buffer) -> float:
    """Penalize left/right leg joints not being mirrored appropriately
    given the half-cycle phase offset. Compares current left-leg pose to
    the right-leg pose recorded half a gait cycle ago (and vice versa)."""
    if left_leg_pose_buffer is None or right_leg_pose_buffer is None:
        return 0.0
    mirror = np.array([1.0, -1.0, 1.0, 1.0, 1.0])  # yaw, roll, pitch, knee, ankle
    left_now = leg_joint_pos[0:5]
    right_now = leg_joint_pos[5:10]
    target_right = left_leg_pose_buffer * mirror
    target_left = right_leg_pose_buffer * mirror
    err = np.sum((right_now - target_right) ** 2) + np.sum((left_now - target_left) ** 2)
    return float(err)
# Orientation / flat torso
def flat_orientation_l2(projected_gravity: np.ndarray) -> float:
    return float(np.sum(projected_gravity[:2] ** 2))


#  Base height
def base_height_l2(height: float, target_height: float) -> float:
    return float((height - target_height) ** 2)


#  Pelvis velocity tracking (normalized, speed-adaptive tolerance)
def pelvis_vel_tracking_penalty(local_lin_vel_xy: np.ndarray, cmd_xy: np.ndarray) -> float:
    """
    p_v = ||v_p_xy - v_c||^2 / max(0.12, 0.5 * ||v_c||^2)

    Speed-dependent tolerance: floor of 0.12 at low/zero commanded speed
    (prevents exploding when standing still), scales with 0.5*||v_c||^2 at
    higher commanded speed so the penalty isn't harsher than necessary.
    """
    err_sq = np.sum((local_lin_vel_xy - cmd_xy) ** 2)
    denom = max(0.12, 0.5 * np.sum(cmd_xy ** 2))
    return float(err_sq / denom)

#  Lateral foot spread
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

# 12. Joint-space penalties
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
#  Gait phase signal
def phase_vector(t: float, period: float) -> np.ndarray:
    phase = np.fmod(t, period) / period
    ang = 2 * np.pi * phase
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)

#  Quaternion -> yaw (needed by envm.py's reset()/_compute_reward(), was missing)
def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))

#  Heading / straight-line walking
def _wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)

def bad_orientation(projected_gravity: np.ndarray, tilt_limit: float) -> bool:
    """Terminate if the robot's projected gravity indicates excessive tilt."""
    tilt = np.linalg.norm(projected_gravity[:2])
    return bool(tilt > tilt_limit)
#####################################################################################

# 14. Roll/pitch angular velocity + vertical bobbing
def ang_vel_xy_l2(ang_vel_b: np.ndarray) -> float:
    return float(np.sum(ang_vel_b[:2] ** 2))

def lin_vel_z_l2(local_lin_vel_z: float) -> float:
    return float(local_lin_vel_z ** 2)

# 11. Action smoothness
def action_rate_l2(action: np.ndarray, prev_action: np.ndarray) -> float:
    return float(np.sum((action - prev_action) ** 2))


def action_smoothness2_l2(action: np.ndarray, prev_action: np.ndarray, prev_prev_action: np.ndarray) -> float:
    """Second-order smoothness: penalizes acceleration in action space."""
    return float(np.sum((action - 2 * prev_action + prev_prev_action) ** 2))

# 15. Survival
def alive_cost() -> float:
    return 1.0

REWARD_WEIGHTS = {
    # tracking
    "track_lin_vel_xy_exp": 0.3,
    "track_ang_vel_z_exp": 0.8,
    "forward_progress": 3.0,

    # heading / straight-line
    "heading_drift": -2.0,
    "lateral_path_deviation": -5.0,
    "yaw_penalty":2.0,

    # gait
    # "gait_phase_contact": 0.8,         
    "gait_phase_tracking": 0.8,
    "feet_air_time_reward": 1.6,
    "symmetry": -0.9,

    # base stability
    "flat_orientation_l2": -1.5,
    "base_height_l2": -1.0,
    # "lin_vel_z_l2": -2.0,
    #"ang_vel_xy_l2": -0.05,

    "pelvis_vel_tracking": -5.0,
    "lateral_spread": -15.0,

  
    "gait_phase_contact": 0.8, 
    "joint_pos_limits": -1.0,
    "joint_penalty": -0.002,
    "joint_vel": -0.001,
    "joint_acc": -1.0e-6,
    "torque": -0.0001,
    "action_rate_l2": -0.02,
    "action_smoothness2_l2": -0.01,

    # survival / termination
    "alive_cost": 1.0,
    "is_terminated": -25.0,
}


def compute_reward(e):
    info = {}
    total = 0.0

    terms = {
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp(e.local_lin_vel[:2], e.commands[:2], std=0.06),
        "track_ang_vel_z_exp": track_ang_vel_z_exp(e.local_ang_vel[2], e.commands[2], std=0.06),
        
        "heading_drift": heading_drift_penalty(e.base_yaw, e.spawn_yaw),
        "lateral_path_deviation": lateral_path_deviation_penalty(e.base_pos[:2], e.spawn_xy, e.spawn_yaw),
        "yaw_penalty": yaw_penalty(e.local_ang_vel[2], e.commands[2]),
        
        "gait_phase_tracking": gait_phase_tracking_reward(e.phase_left, e.phase_right,
                                                            e.foot_contact[0], e.foot_contact[1]),
        "feet_air_time_reward": feet_air_time_reward(e.foot_touchdown_event, e.foot_air_time,
                                                       e.reward_cfg.target_feet_air_time, e.commands[:2]),
        "symmetry": symmetry_penalty(e.leg_joint_pos, e.left_leg_pose_buffer, e.right_leg_pose_buffer),

        "flat_orientation_l2": flat_orientation_l2(e.projected_gravity),
        "base_height_l2": base_height_l2(e.base_pos[2], e.reward_cfg.target_base_height),
        "lin_vel_z_l2": lin_vel_z_l2(e.local_lin_vel[2]),
        "ang_vel_xy_l2": ang_vel_xy_l2(e.local_ang_vel),
        "pelvis_vel_tracking": pelvis_vel_tracking_penalty(e.local_lin_vel[:2], e.commands[:2]),
        "lateral_spread": lateral_spread_penalty(e.data.body("left_foot").xpos, e.data.body("right_foot").xpos),

        "joint_pos_limits": joint_pos_limits(e.joint_pos, e.joint_lower, e.joint_upper),
        "joint_penalty": joint_penalty(e.leg_joint_pos, e.default_leg_joint_pos),
        "joint_vel": joint_vel_penalty(e.leg_joint_vel),
        "joint_acc": joint_acc_penalty(e.leg_joint_vel, e.prev_leg_joint_vel, e.dt_control),
        "torque": torque_penalty(e.last_actuator_force, e.qvel_actuated),
        "action_rate_l2": action_rate_l2(e.action, e.prev_action),
        "action_smoothness2_l2": action_smoothness2_l2(e.action, e.prev_action, e.prev_prev_action),

        "alive_cost": alive_cost(),
    }

    for name, raw in terms.items():
        weight = REWARD_WEIGHTS.get(name, 0.0)
        weighted = weight * raw
        info[f"rew/{name}"] = weighted
        total += weighted


    return total, info