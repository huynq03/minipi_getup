"""Hardware-consistent joint-position action with a torque monitor.

The target, ``q_default + scale * a_k`` (absolute, the paper's form, the default) or
``q(t_k) + scale * a_k`` (relative), is clamped to the joint limits. It is latched
once per policy step and held for all physics substeps, as a motor driver holding a
50 Hz position target would. (The baseline task's
``SettleRelativeJointPositionAction`` re-adds the offset to the current position at
every 2 ms substep, which keeps a constant P push that real hardware wouldn't
produce.) The operational effort limit caps the PD torque.

While a freshly reset env settles (``settle_steps``, marked by the reset event in
``env.extras["settle_mask"]``), the target holds the current pose and the policy
action is ignored.

``TorqueMonitor`` records every substep's joint-space actuator torques
(``qfrc_actuator``) into a histogram. They are stale by one substep, so a step's
window is its 10 substeps shifted by one. The histogram gives mean / p95 / p99 / max /
saturation statistics without storing every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import (
  RelativeJointPositionAction,
  RelativeJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TorqueMonitor:
  """Histogram-based accumulator of |tau| samples (env x joint x substep)."""

  def __init__(self, num_joints: int, limit: float, device, max_torque: float = 20.0):
    self.limit = limit
    self.bin_width = 0.02
    self.num_bins = int(max_torque / self.bin_width)
    self.max_torque = max_torque
    self._device = device
    self._num_joints = num_joints
    self.mask: torch.Tensor | None = None
    self.clear()

  def clear(self) -> None:
    d = self._device
    self.hist = torch.zeros(self.num_bins, device=d, dtype=torch.float64)
    self.count = 0.0
    self.total = torch.zeros((), device=d, dtype=torch.float64)
    self.saturated = torch.zeros((), device=d, dtype=torch.float64)
    self.joint_max = torch.zeros(self._num_joints, device=d)
    self.power_total = torch.zeros((), device=d, dtype=torch.float64)
    self.power_count = 0.0

  def add(self, tau: torch.Tensor, qvel: torch.Tensor) -> None:
    if self.mask is not None:
      tau = tau[self.mask]
      qvel = qvel[self.mask]
    a = tau.abs()
    idx = (a / self.bin_width).long().clamp_(max=self.num_bins - 1)
    self.hist += torch.bincount(idx.flatten(), minlength=self.num_bins).double()
    self.count += a.numel()
    self.total += a.sum().double()
    self.saturated += (a >= 0.9 * self.limit).sum().double()
    if a.numel() > 0:
      self.joint_max = torch.maximum(self.joint_max, a.amax(dim=0))
    self.power_total += (tau * qvel).abs().sum(dim=-1).sum().double()
    self.power_count += tau.shape[0]

  def quantile(self, q: float) -> float:
    if self.count == 0:
      return 0.0
    cdf = torch.cumsum(self.hist, 0) / self.count
    i = int(
      torch.searchsorted(cdf, torch.tensor(q, device=cdf.device, dtype=cdf.dtype))
    )
    return (i + 1) * self.bin_width

  def summary(self) -> dict[str, float]:
    if self.count == 0:
      return {}
    return {
      "torque_mean": float(self.total / self.count),
      "torque_p95": self.quantile(0.95),
      "torque_p99": self.quantile(0.99),
      "torque_max": float(self.joint_max.max()),
      "torque_sat_frac": float(self.saturated / self.count),
      "mech_power_mean": float(self.power_total / max(self.power_count, 1.0)),
    }


@dataclass(kw_only=True)
class FtsrJointPositionActionCfg(RelativeJointPositionActionCfg):
  settle_steps: int = 0
  """Env steps after a marked reset during which the current pose is held."""

  relative: bool = True
  """True: q_target = q + scale * a. False: q_target = q_default + scale * a (paper)."""

  operational_torque_limit: float = 9.0
  """Effort envelope used for the saturation statistic (|tau| >= 0.9 x limit)."""

  max_action_step: float = 0.0
  """If > 0: the action moves at most this far per policy step from the previous
  (limited) action, and the settle hold forces it to 0. ``raw_action`` is the limited
  action, so ``last_action(action_name=...)`` observes what was executed. Unlike
  ``max_target_step`` this needs no state beyond ``last_action``, so the same clamp
  can run inside the exported policy (deployment feeds back its raw output, starting
  from 0). 0 disables it."""

  max_target_step: float = 0.0
  """If > 0: the joint target moves at most this far (rad) per policy step, from the
  previous target (from the measured pose after a reset). A hard rate limit that the
  deployment wrapper must apply as well. 0 disables it."""

  def build(self, env: ManagerBasedRlEnv) -> FtsrJointPositionAction:
    return FtsrJointPositionAction(self, env)


class FtsrJointPositionAction(RelativeJointPositionAction):
  cfg: FtsrJointPositionActionCfg

  def __init__(self, cfg: FtsrJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._target = torch.zeros_like(self._raw_actions)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.monitor = TorqueMonitor(
      len(self._target_ids), cfg.operational_torque_limit, env.device
    )

  def _settling(self) -> torch.Tensor:
    in_window = self._env.episode_length_buf < self.cfg.settle_steps
    return in_window & self._env.extras.get("settle_mask", in_window)

  def process_actions(self, actions: torch.Tensor) -> None:
    if self.cfg.max_action_step > 0:
      prev = self._raw_actions  # Previous limited action; 0 after a reset.
      step = self.cfg.max_action_step
      actions = torch.clamp(actions, min=prev - step, max=prev + step)
      if self.cfg.settle_steps > 0:
        actions = torch.where(
          self._settling().unsqueeze(-1), torch.zeros_like(actions), actions
        )
    super().process_actions(actions)
    q = self._entity.data.joint_pos[:, self._target_ids]
    if self.cfg.relative:
      target = q + self._processed_actions
    else:
      default = self._entity.data.default_joint_pos[:, self._target_ids]
      target = default + self._processed_actions
    if self.cfg.settle_steps > 0:
      target = torch.where(self._settling().unsqueeze(-1), q, target)
    if self.cfg.max_target_step > 0:
      prev = torch.where(self._fresh.unsqueeze(-1), q, self._target)
      step = self.cfg.max_target_step
      target = torch.clamp(target, min=prev - step, max=prev + step)
    self._fresh[:] = False
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._target = torch.clamp(target, min=limits[..., 0], max=limits[..., 1])

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def apply_actions(self) -> None:
    data = self._entity.data
    self.monitor.add(data.qfrc_actuator, data.joint_vel)
    self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)
