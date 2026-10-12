"""Configuration delta for staged recovery on top of host_safe_recovery.

Removes joint-specific strategy shaping (hip-roll opening, knee flexion) and adds
task-space terms that follow: legs in -> feet down -> COM over support -> rise ->
default posture. See REWARDS.md for the equations and activation conditions.
"""

from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg, SafeRecoveryPPOCfg


class StagedRecoveryCfg(SafeRecoveryCfg):
  class safe_recovery(SafeRecoveryCfg.safe_recovery):
    # Hardware velocity requirement: |qd| < 6 rad/s.
    evaluation_ceiling_rad_s = 6.0

  class staged:
    # Motor torque-speed limit (accelerating torque fades to 0 at the max speed).
    motor_max_speed_rad_s = 6.0
    motor_speed_band_rad_s = 1.5
    # Absolute reset half-spans = 25% of the baseline ROM spans, so the reset
    # distribution does not widen with the 120-degree hip-roll limit.
    reset_half_span = {
      "hip_pitch": 0.75,
      "hip_roll": 0.4227,
      "thigh": 0.225,
      "calf": 0.575,
      "ankle_pitch": 0.45,
      "ankle_roll": 0.275,
    }
    # Support polygon: corners of a loaded sole within this height of the ground.
    corner_contact_height_m = 0.012
    support_foot_min = 0.15
    com_support_sigma_m = 0.03
    com_centre_sigma_m = 0.06
    # Torso off the floor (sitting, kneeling, squatting and above).
    lift_gate_height_center = 0.14
    lift_gate_height_width = 0.03
    # Legs in: feet within this horizontal radius of the COM projection.
    feet_under_com_radius_m = 0.12
    feet_under_com_scale_m = 0.08
    # Stance: lateral clearance between sole footprints (zero joints: 0.10 m) and
    # sole centre distance (zero joints: 0.16 m).
    stance_min_gap_m = 0.04
    stance_max_width_m = 0.26
    stance_gap_scale_m = 0.03
    stance_width_scale_m = 0.08
    stance_cost_cap = 4.0
    # Height credit keeps this fraction when the COM is not over the support.
    height_credit_com_floor = 0.4
    unsupported_ascent_threshold_m_s = 0.05
    unsupported_ascent_scale_m_s = 0.20
    # Reverse knee: calf < 0 bends the knee backwards toward the -0.65 stop.
    knee_reverse_soft_rad = 0.25
    knee_reverse_span_rad = 0.40
    # Knee support: brief use is cheap, sustained use costs the full penalty.
    knee_contact_support_min = 0.10
    knee_contact_ramp_s = 0.8
    knee_contact_initial_fraction = 0.25
    knee_contact_decay_s = 0.3
    # Stage events (monitoring only, no reward gate).
    stage_feet_support = 0.3
    stage_com_score = 0.5
    # Rise onset = last upward crossing of this base height before first standing.
    stage_rise_height_m = 0.24
    stage_rise_upright = 0.5
    stage_com_supported_m = 0.01

  class constraints(SafeRecoveryCfg.constraints):
    class scales(SafeRecoveryCfg.constraints.scales):
      # No joint-specific strategy rewards: removed.
      target_loaded_knee_flexion = 0.0
      target_knee_sole_progress = 0.0
      target_recovery_hip_roll = 0.0
      style_loaded_hip_roll_excess = 0.0
      style_loaded_knee_hyperextension = 0.0
      # Pipeline terms (raw scales, multiplied by dt = 0.02 in HoST).
      target_feet_under_com = 3.0
      target_two_foot_flat_support = 4.0
      target_com_support = 6.0
      style_stance_width = -4.0
      style_unsupported_ascent = -6.0
      style_knee_reverse_lock = -8.0
      # Overspeed cost (excess over 4 rad/s, on the substep peak), moved from the
      # regu group (group weight 0.1) to style (1.0) and raised for |qd| < 6 rad/s.
      regu_joint_overspeed = 0.0
      style_joint_overspeed = -1.0


class StagedRecoveryPPOCfg(SafeRecoveryPPOCfg):
  pass
