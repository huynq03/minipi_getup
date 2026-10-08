"""v2 = v1 config plus two intentional changes (see rewards.py):

1. Stage 3 entry (``ready``) tolerates joint overshoot up to ``stage3_entry_overshoot``.
   Strict stable success keeps v1's ``joint_overshoot_tolerance`` (0.015 rad).
2. ``target_joint_margin``: a positive, standing-gated joint-margin reward in the
   additive ``target`` group.

Everything else (plant, controller, observations, PPO, curricula, DR, Stage 1/2,
no-roll/no-prone shaping, strict success) is inherited unchanged from v1.
"""

from dataclasses import dataclass

from minipi_getup.host_ftsr_stage.config import StageCfg, StagePPOCfg, StageSettings


@dataclass(frozen=True)
class StageSettingsV2(StageSettings):
  # Learning-stage entry tolerance only (observed v1 standing overshoot: 0.023 rad).
  # Not a hardware tolerance; strict success still uses joint_overshoot_tolerance.
  stage3_entry_overshoot: float = 0.03
  # Interior band below each physical limit. Feasible standing poses (zero pose and
  # the IK support poses at >= 0.9 H) stay >= 0.12 rad from every limit, so they
  # score 1; only the deepest IK crouch touches the band, and crouches are gated out.
  joint_margin_rad: float = 0.05
  # Per-joint factor exp(-(d / sigma)^2): 0.37 at the limit, 0.12 at 0.023 rad beyond.
  joint_margin_sigma: float = 0.05
  # Smooth standing-progress gate (late Stage 2 and Stage 3): upright, high, loaded.
  margin_gate_up: tuple[float, float] = (0.70, 0.95)
  margin_gate_height: tuple[float, float] = (0.75, 0.92)  # fractions of H
  margin_gate_load: tuple[float, float] = (0.30, 0.55)  # total foot load / weight


class StageCfgV2(StageCfg):
  class constraints(StageCfg.constraints):
    class scales(StageCfg.constraints.scales):
      # Positive, in [0, 1] per second before dt scaling; v1's standing term peaks
      # at standing_reward = 4, so a fully standing robot with good stance still
      # earns most of its target return from target_stage_standing.
      target_joint_margin = 1.0


class StagePPOCfgV2(StagePPOCfg):
  class runner(StagePPOCfg.runner):
    run_name = "host_ftsr_stage_v2"
