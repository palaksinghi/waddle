import numpy as np

def phase_vector(t: float, period: float) -> np.ndarray:
    phase = np.fmod(t, period) / period
    ang = 2 * np.pi * phase
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)


def quat_to_yaw(quat: np.ndarray) -> float:
    """quat = [w, x, y, z] -> yaw angle (rotation about world z-axis)."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def bad_orientation(projected_gravity: np.ndarray, limit_angle: float) -> bool:
    cos_tilt = -projected_gravity[2]
    tilt_angle = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
    return bool(tilt_angle > limit_angle)


def track_lin_vel_xy_exp(lin_vel_xy: np.ndarray, cmd_xy: np.ndarray, std: float) -> float:
    err = np.sum((cmd_xy - lin_vel_xy) ** 2)
    return float(np.exp(-err / std**2))


def track_ang_vel_z_exp(ang_vel_z: float, cmd_z: float, std: float) -> float:
    err = (cmd_z - ang_vel_z) ** 2
    return float(np.exp(-err / std**2))


def heading_penalty(current_yaw: float, initial_yaw: float, cmd_wz: float, turn_threshold: float = 0.05) -> float:
    if abs(cmd_wz) > turn_threshold:
        return 0.0
    err = (current_yaw - initial_yaw) ** 2
    return float(err)


def gait_phase_contact_reward(t, period, left_contact: bool, right_contact: bool) -> float:
    ph = phase_vector(t, period)
    left_should_contact = ph[0] >= 0.0
    right_should_contact = not left_should_contact
    left_match = float(left_contact == left_should_contact)
    right_match = float(right_contact == right_should_contact)
    return 0.5 * (left_match + right_match)


def feet_air_time(last_air_time: np.ndarray, first_contact: np.ndarray, cmd_xy: np.ndarray, threshold=0.05) -> float:
    air_time_bonus = np.clip(last_air_time - threshold, a_min=0.0, a_max=None)
    reward = np.sum(air_time_bonus * first_contact.astype(np.float32))
    if np.linalg.norm(cmd_xy) <= 0.1:
        reward = 0.0
    return float(reward)


def flat_orientation_l2(projected_gravity: np.ndarray) -> float:
    return float(np.sum(projected_gravity[:2] ** 2))


def base_height_l2(height: float, target_height: float) -> float:
    return float((height - target_height) ** 2)


# NEW -- envm.py calls this but it didn't exist anywhere yet, this was
# the second crash waiting to happen right after the REWARD_WEIGHTS one.
# Rewards forward displacement along the commanded xy direction since
# the last step. Keep it small-weighted -- it's a dense shaping term
# that can be gamed (e.g. shuffling forward while unstable), so it's
# a *supplement* to track_lin_vel_xy_exp, not a replacement.
def forward_progress(base_pos_xy: np.ndarray, prev_base_pos_xy: np.ndarray) -> float:
    delta = np.asarray(base_pos_xy, dtype=np.float32) - np.asarray(prev_base_pos_xy, dtype=np.float32)
    return float(delta[0])  # +x displacement this step


def action_rate_l2(action: np.ndarray, prev_action: np.ndarray) -> float:
    return float(np.sum((action - prev_action) ** 2))


def joint_pos_limits(joint_pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    out = -np.clip(joint_pos - lower, a_min=None, a_max=0.0)
    out += np.clip(joint_pos - upper, a_min=0.0, a_max=None)
    return float(np.sum(out))


def ang_vel_xy_l2(ang_vel_b: np.ndarray) -> float:
    return float(np.sum(ang_vel_b[:2] ** 2))


def alive_cost() -> float:
    # kept as a cost (matches envm.py's key name "alive_cost" and the
    # weight below is what actually turns this into a positive or
    # negative contribution to reward)
    return -1.0


# =============================================================================
# FLAT WEIGHTS -- matches _compute_reward()'s exact terms dict keys.
# envm.py does `w = R.REWARD_WEIGHTS` and does `w[k] * v` for every key
# in `terms`, so EVERY key below is required -- missing even one will
# throw a KeyError the same way REWARD_WEIGHTS itself just did.
#
# These are STAND-phase-leaning weights (velocity tracking kept low,
# balance/orientation kept high) since env.reset() currently forces a
# constant forward command (cmd[0] = 0.2) rather than sampling 0 -- see
# note below the dict about that.
# =============================================================================
REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 0.0,
    "heading": 0.5,
    "gait_phase_contact": 0.0,
    "feet_air_time": 2.0,
    "flat_orientation_l2": -5.0,
    "base_height_l2": -3.0,
    "forward_progress": 2.0,
    "action_rate_l2": -0.02,
    "joint_pos_limits": -1.0,
    "ang_vel_xy_l2": -0.15,
    "alive_cost": 0.5,          # positive weight * negative alive_cost() = -0.5/step baseline;
                                  # see note below if you want a true survival bonus instead
    "is_terminated": -8.0,
}

# -----------------------------------------------------------------------
# HEADS-UP, separate from the crash fix:
# In envm.py's reset(), self.cmd gets overwritten right after the random
# sample to a FIXED forward_vel=0.2 every single episode:
#
#     self.cmd = self.np_random.uniform(...)   # <- this line is dead code
#     FORWARD_VEL = 0.2
#     self.cmd = np.zeros(3, dtype=np.float32)
#     self.cmd[0] = FORWARD_VEL                # <- overwrites it
#
# So right now there is no STAND-only phase possible at the env level --
# every episode commands 0.2 m/s forward from step 0, even while the
# robot is still learning to balance. That's the same conflict from
# before (balance vs. velocity fighting), just moved from the reward
# weights into the command sampling. Once training is stable again,
# consider setting cmd[0] = 0.0 for the first N episodes/timesteps
# (a simple counter in __init__ works) before ramping to 0.2 -- that
# is the actual STAND -> WALK curriculum, at the command level.
# -----------------------------------------------------------------------