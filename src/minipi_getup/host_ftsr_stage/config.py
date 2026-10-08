"""Only intentional reward changes; controller/observations/PPO remain HoST-real."""

from dataclasses import dataclass

from minipi_getup.host.config import PiCfg, PiCfgPPO
from minipi_getup.host.train_clpai import REAL_KD, REAL_KP

from .geometry import XML


@dataclass(frozen=True)
class StageSettings:
  erect_dwell_s: float = 0.12
  support_dwell_s: float = 0.20
  blend_s: float = 0.40
  hold_s: float = 1.0
  lateral_free_deg: float = 30.0
  stable_tilt_deg: float = 18.0
  foot_min_weight_fraction: float = 0.08
  total_min_weight_fraction: float = 0.55
  joint_overshoot_tolerance: float = 0.015
  stable_pose_max_error: float = 0.45
  stable_symmetry_rms: float = 0.20
  early_regularization_fraction: float = 0.10
  erection_progress_weight: float = 2.0
  support_progress_weight: float = 2.0
  lateral_cost: float = 0.15
  prone_cost: float = 0.25
  twist_cost: float = 0.03
  hard_limit_cost: float = 0.30
  unfinished_cost: float = 0.03
  standing_reward: float = 4.0


class StageCfg(PiCfg):
  class asset(PiCfg.asset):
    file = str(XML)

  class control(PiCfg.control):
    stiffness = dict(REAL_KP)
    damping = dict(REAL_KD)

  # Keep four critics, group weights, original positive task product, all regu.
  # Replace old geometry-insensitive style/target by the staged terms.
  class constraints(PiCfg.constraints):
    class scales(PiCfg.constraints.scales):
      style_hip_yaw_deviation = 0
      style_hip_roll_deviation = 0
      style_left_foot_displacement = 0
      style_right_foot_displacement = 0
      style_knee_deviation = 0
      style_ground_parallel = 0
      style_feet_distance = 0
      style_style_ang_vel_xy = 0
      target_ang_vel_xy = 0
      target_lin_vel_xy = 0
      target_feet_height_var = 0
      target_target_orientation = 0
      target_target_base_height = 0
      style_stage_shaping = 1.0
      target_stage_standing = 1.0


class StagePPOCfg(PiCfgPPO):
  class runner(PiCfgPPO.runner):
    run_name = "host_ftsr_stage_v1"
    save_interval = 50
