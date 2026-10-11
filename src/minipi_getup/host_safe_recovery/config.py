"""Configuration delta for HoST-only safe recovery.

The actor, critics, observations, PPO, action law, and original rewards are inherited
unchanged. This delta adds the physical controller, a 30-to-0 N pull curriculum, the
three safety objectives, and foot-supported recovery shaping.
"""

from minipi_getup.host.config import PiCfg, PiCfgPPO
from minipi_getup.host_safe_recovery.asset import REAL_KD, REAL_KP


class SafeRecoveryCfg(PiCfg):
  class control(PiCfg.control):
    stiffness = dict(REAL_KP)
    damping = dict(REAL_KD)

  class curriculum(PiCfg.curriculum):
    force = 30

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
    force_curriculum_step_n = 10.0

    # Contact-force shaping is normalized by each randomized robot's weight.
    contact_force_deadband_weight_fraction = 0.02
    contact_force_scale_weight_fraction = 0.25
    support_transfer_height_center = 0.20
    support_transfer_height_width = 0.05
    support_transfer_upright_center = 0.35
    support_transfer_upright_width = 0.15
    foot_height_center = 0.18
    foot_height_width = 0.06
    support_rise_time_s = 0.10
    support_decay_time_s = 0.02
    upward_velocity_soft_m_s = 0.35
    knee_support_height_center = 0.18
    knee_support_height_width = 0.04
    knee_lift_velocity_scale_m_s = 0.20

  class constraints(PiCfg.constraints):
    class scales(PiCfg.constraints.scales):
      # Config scales are multiplied by dt by the unchanged HoST environment.
      regu_joint_overspeed = -0.10
      target_default_posture = -2.0
      target_stable_standing = -0.05
      # Raw foot scores are bounded: load <= 1.5, supported height <= 1.5,
      # non-foot support <= 1.0. HoST multiplies these scales by dt=0.02.
      target_foot_load_support = 8.0
      style_nonfoot_support = -15.0
      target_foot_supported_height = 8.0
      # Style critic weight is 1: strong cost for ballistic ascent, including
      # takeoff after established support (20% residual cost).
      style_ballistic_ascent = -80.0
      # Additional knee-only load cost; torso support remains unrestricted.
      style_knee_load_support = -45.0


class SafeRecoveryPPOCfg(PiCfgPPO):
  pass
