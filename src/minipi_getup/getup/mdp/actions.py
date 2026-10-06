"""Action terms for the getup task."""

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


@dataclass(kw_only=True)
class SettleRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
  """RelativeJointPositionActionCfg that disables actions for the first N steps.

  Since the robot is dropped from a height in a random configuration, actions are
  suppressed until ``settle_steps`` env steps have passed, allowing the robot to land
  and settle before the policy takes over.
  """

  settle_steps: int = 0
  """Number of env steps after reset during which the policy action is ignored and the
  robot holds its current position. Set to 0 to disable."""

  clip_target_to_limits: bool = False
  """Clamp the PD position target to the joint position limits, so the policy cannot
  command a joint into its hard stop (as a hardware controller would)."""

  def build(self, env: ManagerBasedRlEnv) -> SettleRelativeJointPositionAction:
    return SettleRelativeJointPositionAction(self, env)


class SettleRelativeJointPositionAction(RelativeJointPositionAction):
  """RelativeJointPositionAction that disables actions for the first N steps."""

  def __init__(
    self,
    cfg: SettleRelativeJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)
    self._settle_steps = cfg.settle_steps
    self._clip_target_to_limits = cfg.clip_target_to_limits

  def apply_actions(self) -> None:
    current_pos = self._entity.data.joint_pos[:, self._target_ids]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = current_pos + self._raw_actions * self._scale - encoder_bias
    if self._settle_steps > 0:
      in_window = self._env.episode_length_buf < self._settle_steps
      was_fallen = self._env.extras.get("settle_mask", in_window)
      settling = (in_window & was_fallen).unsqueeze(-1)
      target = torch.where(settling, current_pos - encoder_bias, target)
    if self._clip_target_to_limits:
      limits = self._entity.data.soft_joint_pos_limits
      assert limits is not None
      limits = limits[:, self._target_ids]
      target = torch.clamp(target, min=limits[..., 0], max=limits[..., 1])
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
