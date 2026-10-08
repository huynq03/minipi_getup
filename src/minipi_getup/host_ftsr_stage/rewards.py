"""Per-environment FTSR-inspired shaping, with no actor observation additions."""

import math

import torch

from .config import StageSettings


def smooth(x, lo, hi):
  t = ((x - lo) / (hi - lo)).clamp(0, 1)
  return t * t * (3 - 2 * t)


def orientation(gravity):
  # Verified from XML root rotation: local +X points upward when supine.
  # This lateral inclination stays zero for a pure sagittal sit-up, even at 90deg pitch.
  up = -gravity[:, 2]
  side = torch.asin(gravity[:, 1].abs().clamp(0, 1))
  prone = (gravity[:, 0] > 0.7) & (up.abs() < 0.55)
  return up, side, prone


class StageLogic:
  def __init__(self, n, device, dt, geometry, cfg=None):
    self.cfg = cfg or StageSettings()
    self.dt = dt
    self.geometry = geometry
    self.stage = torch.zeros(n, device=device, dtype=torch.long)
    self.dwell = torch.zeros(n, device=device)
    self.blend = torch.zeros(n, 2, device=device)
    self.hold = torch.zeros(n, device=device)
    self.upright_hold = torch.zeros(n, device=device)
    self.highwater = torch.zeros(n, 2, device=device)
    self.started = torch.zeros(n, device=device, dtype=torch.bool)
    self.ever_stable = torch.zeros_like(self.started)
    self.prone_seen = torch.zeros_like(self.started)
    self.side_seen = torch.zeros_like(self.started)
    self.last_side = torch.zeros_like(self.started)
    self.last_prone = torch.zeros_like(self.started)
    self.previous_ready = torch.zeros_like(self.started)
    self.transitions = torch.zeros(n, device=device)
    self.regressions = torch.zeros(n, device=device)
    self.roll_events = torch.zeros(n, device=device)
    self.time_in_stage = torch.zeros(n, 3, device=device)
    self.max_side = torch.zeros(n, device=device)
    self.last = {}

  def reset(self, ids):
    for name in (
      "stage",
      "dwell",
      "blend",
      "hold",
      "upright_hold",
      "highwater",
      "started",
      "ever_stable",
      "prone_seen",
      "side_seen",
      "last_side",
      "last_prone",
      "previous_ready",
      "transitions",
      "regressions",
      "roll_events",
      "time_in_stage",
      "max_side",
    ):
      getattr(self, name)[ids] = 0

  def update(self, f, active):
    c = self.cfg
    dt = self.dt
    H = self.geometry["standing_height"]
    up, side, prone = orientation(f["gravity"])
    upright = (
      (up > math.cos(math.radians(c.stable_tilt_deg)))
      & (f["height"] > 0.92 * H)
      & active
    )
    self.upright_hold.copy_(
      torch.where(upright, self.upright_hold + dt, torch.zeros_like(self.upright_hold))
    )
    erect = (up > 0.5) & (f["height"] > 0.65 * min(self.geometry["support_heights"]))
    both = (f["foot_load"] > c.foot_min_weight_fraction).all(-1)
    support = both & (f["foot_load"].sum(-1) > c.total_min_weight_fraction)
    ready = (
      (up > math.cos(math.radians(20)))
      & (f["height"] > 0.92 * H)
      & support
      & (f["foot_up"].min(-1).values > 0.9)
      & (f["overshoot"] < c.joint_overshoot_tolerance)
    )
    old = self.stage.clone()
    eligible = torch.where(old == 0, erect, ready) & (old < 2) & active
    self.dwell.copy_(
      torch.where(eligible, self.dwell + dt, torch.zeros_like(self.dwell))
    )
    threshold = torch.where(old == 0, c.erect_dwell_s, c.support_dwell_s)
    advance = (self.dwell >= threshold) & eligible
    self.stage.add_(advance.long())
    self.transitions.add_(advance.float())
    self.dwell[advance] = 0
    # Latch progression. No stage-entry bonus. Lost balance is recoverable using
    # the same geometry shaping; current readiness reduces posture pressure.
    lost = (old == 2) & self.previous_ready & ~ready & active
    self.regressions.add_(lost.float())
    self.previous_ready.copy_(ready)
    target = torch.stack((self.stage >= 1, self.stage >= 2), -1).float()
    self.blend.add_(
      (target - self.blend).clamp(-dt / c.blend_s, dt / c.blend_s) * active[:, None]
    )
    erect_gate = smooth(up, 0.1, 0.85) * smooth(f["height"], 0.12, 0.26)
    support_gate = self.blend[:, 0] * erect_gate
    anatomical = (
      (f["pose_max"] < c.stable_pose_max_error)
      & (f["symmetry_rms"] < c.stable_symmetry_rms)
      & (f["overshoot"] < c.joint_overshoot_tolerance)
    )
    stable = (
      (up > math.cos(math.radians(c.stable_tilt_deg)))
      & (f["height"] > 0.92 * H)
      & (f["height"] < 1.10 * H)
      & support
      & anatomical
      & (f["foot_up"].min(-1).values > 0.94)
      & (f["balance_error"] < 0.025)
      & (f["ang_speed"] < 0.8)
      & (f["lin_speed"] < 0.15)
      & active
    )
    self.hold.copy_(torch.where(stable, self.hold + dt, torch.zeros_like(self.hold)))
    success = self.hold >= c.hold_s
    self.ever_stable.logical_or_(success)
    excessive_side = side > math.radians(60)
    # Hysteresis: count a new excursion only after returning below 40 degrees.
    event = (excessive_side & ~self.last_side) | (prone & ~self.last_prone)
    self.roll_events.add_((event & active).float())
    self.last_side.copy_(
      torch.where(side < math.radians(40), False, self.last_side | excessive_side)
    )
    self.last_prone.copy_(
      torch.where(f["gravity"][:, 0] < 0.4, False, self.last_prone | prone)
    )
    self.prone_seen.logical_or_(prone & active)
    self.side_seen.logical_or_(excessive_side & active)
    self.max_side.copy_(torch.maximum(self.max_side, side * active))
    self.time_in_stage.add_(
      torch.nn.functional.one_hot(self.stage, 3) * active[:, None] * dt
    )
    # Incremental high-water progress is earned at most once per episode.
    # Returning to a crouch/rolling again cannot farm transition bonuses.
    erection = 0.55 * up.clamp(0, 1) + 0.45 * smooth(f["head_height"], 0.08, H + 0.08)
    useful = (
      0.35 * smooth(f["height"], 0.12, H)
      + 0.20 * f["support_pose"]
      + 0.20 * f["placement"]
      + 0.25 * f["foot_load"].clamp(0, 0.5).sum(-1)
    )
    potentials = torch.stack((erection, useful * erect_gate), -1)
    first = active & ~self.started
    self.highwater[first] = potentials[first]
    gains = (potentials - self.highwater).clamp(min=0) * active[:, None]
    self.highwater.copy_(torch.maximum(self.highwater, potentials * active[:, None]))
    self.started.logical_or_(active)
    progress = (
      c.erection_progress_weight * gains[:, 0]
      + c.support_progress_weight * gains[:, 1] * (0.25 + 0.75 * support_gate)
    ) / dt
    lateral = smooth(side, math.radians(c.lateral_free_deg), math.pi / 2)
    prone_cost = smooth(f["gravity"][:, 0], 0.3, 0.9)
    twist = ((f["offaxis_speed"] - 2).clamp(min=0) / 6).clamp(max=1)
    limit_cost = (f["overshoot"] / 0.05).clamp(0, 2)
    symmetry_cost = (
      support_gate * (f["symmetry_rms"] / 0.4).square().clamp(max=1) * 0.10
    )
    shaping = (
      progress
      - c.lateral_cost * lateral
      - c.prone_cost * prone_cost
      - c.twist_cost * twist
      - c.hard_limit_cost * limit_cost
      - c.unfinished_cost * (~stable)
      - symmetry_cost
    )
    # Standing quality is continuously gated; stage index never creates a reward jump.
    stand_quality = (
      smooth(up, 0.8, 0.99)
      * torch.exp(-(((f["height"] - H) / 0.045) ** 2))
      * f["nominal_pose"]
      * f["placement"]
    )
    stand_quality *= (
      f["foot_up"].clamp(0, 1).prod(-1)
      * f["foot_load"].clamp(0, 0.25).prod(-1)
      / (0.25**2)
    )
    stand_quality *= torch.exp(
      -((f["ang_speed"] / 0.8) ** 2)
      - (f["lin_speed"] / 0.2) ** 2
      - (f["balance_error"] / 0.04) ** 2
    )
    stand_quality *= torch.exp(-((f["overshoot"] / 0.015) ** 2))
    standing = (
      c.standing_reward
      * stand_quality
      * (0.5 + 0.5 * (self.hold / c.hold_s).clamp(0, 1))
    )
    self.last = dict(
      style=shaping * active,
      target=standing * active,
      stable=stable,
      success=success,
      up=up,
      side=side,
      prone=prone,
      regularization_gate=c.early_regularization_fraction
      + (1 - c.early_regularization_fraction) * self.blend[:, 1] * erect_gate,
      support_gate=support_gate,
      progress=progress,
      stand_quality=stand_quality,
    )
    return self.last
