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
    # Overspeed cost starts here (was 4): slow, deliberate motion.
    overspeed_soft_rad_s = 3.0

  class staged:
    # Motor torque-speed limit (accelerating torque fades to 0 at the max speed).
    # 4 rad/s (was 6): leaves a 2 rad/s margin below the 6 rad/s hardware limit.
    motor_max_speed_rad_s = 4.0
    motor_speed_band_rad_s = 1.0
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
    # Stance cost while rising (not yet standing), as a fraction of the standing cost.
    stance_rise_fraction = 0.25
    # Standing load split; the floor keeps the ratio finite while unloaded.
    load_imbalance_floor_weight_fraction = 0.1
    # Leg-on-leg contact: 0.5% W deadband, saturating with a 5% W scale (~3.4 N).
    leg_contact_deadband_weight_fraction = 0.005
    leg_contact_scale_weight_fraction = 0.05
    # Revision tag, recorded in the run metadata (config.json).
    revision = "6a"
    # Rising credit = floor + w * support_gate + (1 - floor - w) * ready, with
    # ready = sqrt(coverage_pair * com_score) * support_gate (revision 6a):
    # both soles planted, COM over their support polygon, both feet loaded
    # (weaker foot >= two_foot_full_share of the load).
    height_credit_com_floor = 0.2
    # Intermediate step: credit = 0.2 + 0.3 * support_gate + 0.5 * ready.
    height_credit_two_foot_weight = 0.3
    two_foot_full_share = 0.3
    # Sole coverage is geometric (revision 6a): strip end points within this
    # clearance of the ground, times a foot-load gate 1 - exp(-relu(F - d W) / (s W)).
    sole_coverage_tolerance_m = 0.003
    sole_coverage_load_deadband_weight_fraction = 0.01
    sole_coverage_load_scale_weight_fraction = 0.05
    # Loaded foot translation (revision 6a): horizontal sole-centre speed above the
    # threshold, gated by a load hold that decays with this time constant while
    # the sole stays within the contact clearance, and resets once it is lifted.
    foot_translation_threshold_m_s = 0.03
    foot_translation_scale_m_s = 0.10
    foot_translation_contact_clearance_m = 0.005
    foot_translation_hold_decay_s = 0.25
    # coverage_pair = w * mean(cov_L, cov_R) + (1 - w) * min(cov_L, cov_R).
    coverage_pair_mean_weight = 0.3
    # Torso over the midpoint of the feet, along the robot lateral axis.
    torso_center_sigma_m = 0.08
    # COM inside the two-foot support polygon: full bonus at this margin.
    com_margin_full_m = 0.025
    # Slow rise: upward COM speed above this costs (a 1.5 s rise is ~0.17 m/s).
    rise_speed_threshold_m_s = 0.20
    rise_speed_scale_m_s = 0.10
    rise_speed_cost_cap = 9.0
    # One-foot rise penalty: active from this base height until standing.
    rise_gate_height_center = 0.10
    rise_gate_height_width = 0.02
    unsupported_ascent_threshold_m_s = 0.05
    unsupported_ascent_scale_m_s = 0.20
    # Reverse knee: calf < 0 bends the knee backwards toward the -0.65 stop.
    # Knee geometry: calf 0 = 37 deg flexed (default), -0.65 = straight leg,
    # below -0.65 = bent backwards (asset limit -2.5 = 106 deg). Standing penalizes calf
    # below -0.25 (< ~22 deg flexion), 1.0 at straight, growing when reversed.
    knee_extension_soft_rad = 0.25
    knee_straight_rad = -0.65
    # Actuated |qd| termination (5 ms peak): off until the success curriculum
    # starts (EMA 0.60), then 12 -> 5 rad/s as it completes (EMA 0.85).
    velocity_termination = True  # training only; evaluation/playback disable it
    velocity_termination_start_rad_s = 12.0
    velocity_termination_final_rad_s = 5.0
    knee_extension_span_rad = 0.40
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
      # HoST's hip-roll deviation limit (1.4 rad one leg / 0.9 both) would cap the
      # widened 120-degree ROM; dropped by request.
      style_hip_roll_deviation = 0.0
      # Pipeline terms (raw scales, multiplied by dt = 0.02 in HoST).
      target_feet_under_com = 3.0
      # Revision 6a: planted soles (coverage) and a centred, stable stance.
      target_sole_contact_coverage = 8.0
      style_loaded_sole_coverage_deficit = -10.0
      target_two_foot_flat_support = 6.0
      target_torso_center_over_feet = 4.0
      target_com_support = 6.0
      target_com_support_margin = 2.0
      target_stable_standing = -1.0
      # Do not slide a loaded foot (e.g. dragging it inward after opening the hip
      # roll): reposition by lifting it. Separate from loaded_sole_slip (-6, kept).
      style_loaded_foot_translation = -4.0
      style_stance_width = -4.0
      style_unsupported_ascent = -6.0
      # Getting up must use both feet: any rising step with the load on one foot pays.
      style_one_foot_rise = -20.0
      # Get up slowly: upward COM speed above 0.2 m/s.
      style_rise_speed = -4.0
      # Standing-gated only: straight or reversed knees are free while getting
      # up but must return toward the flexed default once standing.
      style_standing_knee_extension = -16.0
      # Standing on one foot (the other resting on it) escaped every per-foot
      # load-weighted cost; this makes the load split itself costly.
      style_standing_load_imbalance = -6.0
      # Legs pressing on each other once the torso is up.
      style_leg_contact = -6.0
      # Overspeed cost (excess over 4 rad/s, on the substep peak), moved from the
      # regu group (group weight 0.1) to style (1.0) and raised for |qd| < 6 rad/s.
      regu_joint_overspeed = 0.0
      style_joint_overspeed = -2.0


class StagedRecoveryPPOCfg(SafeRecoveryPPOCfg):
  pass
