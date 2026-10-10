"""Configuration delta for HoST-only safe recovery.

The actor, critics, observations, PPO, action law, curricula, and original rewards are
inherited unchanged. Only the physical controller and three requested reward objectives
are added here.
"""

from minipi_getup.host.config import PiCfg, PiCfgPPO
from minipi_getup.host_safe_recovery.asset import REAL_KD, REAL_KP


class SafeRecoveryCfg(PiCfg):
  class control(PiCfg.control):
    stiffness = dict(REAL_KP)
    damping = dict(REAL_KD)

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

  class constraints(PiCfg.constraints):
    class scales(PiCfg.constraints.scales):
      # Config scales are multiplied by dt by the unchanged HoST environment.
      regu_joint_overspeed = -0.10
      target_default_posture = -2.0
      target_stable_standing = -0.05


class SafeRecoveryPPOCfg(PiCfgPPO):
  pass
