"""Reward functions for the getup task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost  # noqa: F401
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Projected gravity in body frame when upright.
_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


def orientation_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for upright orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  up = _UP_VEC.to(gravity.device)
  error = torch.sum(torch.square(up - gravity), dim=-1)
  return torch.exp(-2.0 * error)


def height_reward(
  env: ManagerBasedRlEnv,
  desired_height: float,
  scale_by_uprightness: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for raising a body (or the mean of several bodies) to the desired height.

  With ``scale_by_uprightness``, the reward is multiplied by how much the torso points
  up (-gravity z in the body frame, clamped to [0, 1]), so lifting the base while lying
  or upside down (e.g. a shoulder stand) earns nothing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].mean(dim=-1)
  clamped = torch.clamp(height, max=desired_height)
  reward = (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)
  if scale_by_uprightness:
    reward = reward * (-asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
  return reward


def _is_upright(asset: Entity, orientation_threshold: float) -> torch.Tensor:
  gravity = asset.data.projected_gravity_b
  up = _UP_VEC.to(gravity.device)
  error = torch.sum(torch.square(up - gravity), dim=-1)
  return (error < orientation_threshold).float()


def _is_at_desired_height(
  asset: Entity, desired_height: float, height_tolerance: float
) -> torch.Tensor:
  height = asset.data.root_link_pos_w[:, 2]
  clamped = torch.clamp(height, max=desired_height)
  error = desired_height - clamped
  return (error < height_tolerance).float()


class gated_posture_reward:
  """Reward for returning to default pose, gated on being upright."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: dict[str, float],
    orientation_threshold: float = 0.01,
    min_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del std  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    # Also gated on root height, so sitting upright on the ground earns nothing.
    gate = (
      _is_upright(asset, orientation_threshold)
      * (asset.data.root_link_pos_w[:, 2] >= min_height).float()
    )
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return gate * torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


class getup_success:
  """Binary success metric: 1 once the robot has stood up, 0 otherwise."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._stood_up = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._stood_up[:] = 0.0
    else:
      self._stood_up[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    desired_height: float = 0.275,
    height_tolerance: float = 0.02,
    orientation_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    standing = _is_upright(asset, orientation_threshold) * _is_at_desired_height(
      asset, desired_height, height_tolerance
    )
    self._stood_up = torch.maximum(self._stood_up, standing)
    return self._stood_up


def base_yaw_rate_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base rotation about the world z axis (spinning in place)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_w[:, 2])


def base_yaw_rate_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize world-z rotation throughout getup, including slow full turns.

  With reward time scaling, the episode cost is proportional to total angular
  travel rather than becoming smaller when the same turn is performed slowly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_w[:, 2].abs()


def upright_base_lin_vel_xy_l2(
  env: ManagerBasedRlEnv,
  orientation_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize horizontal base drift (stepping, shuffling) once upright."""
  asset: Entity = env.scene[asset_cfg.name]
  gate = _is_upright(asset, orientation_threshold)
  lin_vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  return gate * torch.sum(torch.square(lin_vel_xy), dim=-1)


def hip_roll_open_when_low(
  env: ManagerBasedRlEnv,
  max_height: float,
  lying_gravity_z: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward spreading the legs (hip roll toward its outward limit) while lying low.

  The outward direction of each joint is the side of its range with the larger
  magnitude. Returns the mean opening fraction in [0, 1], gated on the root height
  being below ``max_height`` and on the torso lying (body-frame gravity z within
  ``±lying_gravity_z``; -1 is upright, +1 upside down), so neither a wide sit nor an
  inverted pose is rewarded.
  """
  asset: Entity = env.scene[asset_cfg.name]
  limits = asset.data.soft_joint_pos_limits
  assert limits is not None
  lower = limits[:, asset_cfg.joint_ids, 0]
  upper = limits[:, asset_cfg.joint_ids, 1]
  outward_limit = torch.where(lower.abs() > upper.abs(), lower, upper)
  opening = (asset.data.joint_pos[:, asset_cfg.joint_ids] / outward_limit).clamp(
    0.0, 1.0
  )
  gate = (asset.data.root_link_pos_w[:, 2] < max_height).float() * (
    asset.data.projected_gravity_b[:, 2].abs() < lying_gravity_z
  ).float()
  return gate * opening.mean(dim=-1)


def upward_speed_when_low(
  env: ManagerBasedRlEnv,
  max_height: float = 0.25,
  max_speed: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize rapid upward launches during preparation, allowing a slow rise."""
  asset: Entity = env.scene[asset_cfg.name]
  excess_speed = (asset.data.root_link_lin_vel_w[:, 2] - max_speed).clamp_min(0.0)
  gate = (asset.data.root_link_pos_w[:, 2] < max_height).float()
  return gate * excess_speed.square()


def feet_under_base(
  env: ManagerBasedRlEnv,
  std: float,
  forward_offset: float,
  upright_gravity_z: float = -0.7,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the feet under the hips once the torso is upright-ish.

  Measures each foot's offset from the base along the base heading and compares it
  with ``forward_offset`` (the standing offset). Gated on body-frame gravity z being
  below ``upright_gravity_z`` (-1 is upright), so it is active from sitting through
  squatting to standing, and pulls the legs in from a sit with the legs out in front.
  """
  asset: Entity = env.scene[asset_cfg.name]
  heading = asset.data.heading_w
  forward = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
  base_xy = asset.data.root_link_pos_w[:, :2]
  feet_xy = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]
  dx = torch.sum((feet_xy - base_xy.unsqueeze(1)) * forward.unsqueeze(1), dim=-1)
  reward = torch.exp(-torch.square((dx - forward_offset) / std)).mean(dim=-1)
  gate = (asset.data.projected_gravity_b[:, 2] < upright_gravity_z).float()
  return gate * reward
