"""Configuration delta for HoST-only safe recovery.

The actor, critics, observations, PPO, action law, and original rewards are inherited
unchanged. This delta adds the physical controller, a 40-to-0 N pull curriculum, the
three safety objectives, and foot-supported recovery shaping.
"""

from minipi_getup.host.config import PiCfg, PiCfgPPO
from minipi_getup.host_safe_recovery.asset import REAL_KD, REAL_KP


class SafeRecoveryCfg(PiCfg):
  class control(PiCfg.control):
    stiffness = dict(REAL_KP)
    damping = dict(REAL_KD)

  class curriculum(PiCfg.curriculum):
    force = 40

  class safe_recovery:
    overspeed_soft_rad_s = 4.0
    evaluation_ceiling_rad_s = 5.0
    # Smooth gate begins providing posture/stability gradients before final standing.
    gate_height_center = 0.25
    gate_height_width = 0.045
    gate_upright_center = 0.65
    gate_upright_width = 0.12
    pose_sigma = {
      "hip_pitch": 0.60,
      "hip_roll": 0.25,
      "thigh": 0.25,
      "calf": 0.45,
      "ankle_pitch": 0.35,
      "ankle_roll": 0.25,
    }
    pose_weight = {
      "hip_pitch": 1.0,
      "hip_roll": 1.0,
      "thigh": 1.0,
      "calf": 1.0,
      "ankle_pitch": 0.8,
      "ankle_roll": 0.6,
    }
    stable_joint_velocity_weight = 1.0
    stable_base_angular_velocity_weight = 0.25
    standing_posture_penalty_multiplier = 2.0
    standing_motion_penalty_multiplier = 2.0
    penalty_success_start = 0.60
    penalty_success_full = 0.85
    penalty_success_ema_episodes = 2048.0
    penalty_stand_hold_s = 0.50
    ascent_excess_scale_m_s = 0.50
    initial_upward_velocity_soft_m_s = 0.35
    penalty_initial_rotation_fraction = 0.25
    penalty_initial_regu_fraction = 1.0 / 3.0
    penalty_stand_support_fraction = 0.80
    recovery_rotation_soft_rad_s = 1.5
    recovery_rotation_standing_residual = 0.25
    sole_slip_soft_m_s = 0.03
    sole_slip_scale_m_s = 0.10
    # Outward hip roll above ~34 deg on a loaded leg; standing uses < 0.5 rad.
    loaded_hip_roll_soft_rad = 0.60
    # Knee use while lifting: calf > 0 flexes (verified by kinematics), calf < 0
    # bends the knee backwards toward the -0.65 stop.
    knee_lift_gate_height_center = 0.14
    knee_lift_gate_height_width = 0.03
    knee_flexion_scale_rad = 0.60
    knee_hyperextension_soft_rad = 0.30
    knee_hyperextension_span_rad = 0.35
    # Full-sole goal: a loaded foot should carry load on (nearly) every strip.
    loaded_sole_min_coverage = 0.90
    sole_tilt_sigma_rad = 0.15
    force_curriculum_step_n = 10.0
    action_scale_min = 0.10
    action_scale_finetune_step = 0.005
    action_scale_success_threshold = 0.85

    # Contact-force shaping is normalized by each randomized robot's weight.
    contact_force_deadband_weight_fraction = 0.02
    contact_force_scale_weight_fraction = 0.25
    # Light contact is enough for placement; load bearing has its own reward.
    sole_contact_scale_weight_fraction = 0.005
    knee_contact_progress_scale_rad = 0.04
    support_transfer_height_center = 0.20
    support_transfer_height_width = 0.05
    support_transfer_upright_center = 0.35
    support_transfer_upright_width = 0.15
    foot_height_center = 0.18
    foot_height_width = 0.06
    support_rise_time_s = 0.10
    support_decay_time_s = 0.02
    upward_velocity_soft_m_s = 0.22
    knee_support_height_center = 0.18
    knee_support_height_width = 0.04
    knee_lift_velocity_scale_m_s = 0.20
    knee_height_credit_floor = 0.35

  class constraints(PiCfg.constraints):
    class scales(PiCfg.constraints.scales):
      # Config scales are multiplied by dt by the unchanged HoST environment.
      regu_joint_overspeed = -0.30
      regu_action_rate = -0.03
      regu_smoothness = -0.03
      style_recovery_rotation = -2.0
      # Linear cost per 0.1 m/s excess: dragging one loaded foot costs about its load credit.
      style_loaded_sole_slip = -6.0
      style_loaded_hip_roll_excess = -12.0
      target_loaded_knee_flexion = 4.0
      style_loaded_knee_hyperextension = -4.0
      style_loaded_sole_coverage_deficit = -8.0
      # Per foot this matches the single-foot load credit (0.5 * 8): a fully
      # tilted loaded foot earns about nothing, a flat loaded foot the full credit.
      style_loaded_sole_tilt = -8.0
      # Fourfold stronger attraction to the default joints once the existing
      # smooth standing gate opens; recovery poses remain weakly constrained.
      target_default_posture = -8.0
      target_stable_standing = -0.05
      # Raw foot scores are bounded: load <= 1.5, supported height <= 1.5,
      # non-foot support <= 1.0. HoST multiplies these scales by dt=0.02.
      target_foot_load_support = 8.0
      target_sole_contact_coverage = 4.0
      target_knee_sole_progress = 6.0
      style_nonfoot_support = -15.0
      target_foot_supported_height = 8.0
      # Immediate quadratic + quartic ascent cost, without foot exemption.
      style_ballistic_ascent = -80.0
      # Additional knee-only load cost; torso support remains unrestricted.
      style_knee_load_support = -45.0


class SafeRecoveryPPOCfg(PiCfgPPO):
  pass
