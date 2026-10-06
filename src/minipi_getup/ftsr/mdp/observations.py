"""FTSR observation terms (privileged teacher inputs and critic extras)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from minipi_getup.ftsr.mdp.assistance import get_wrench

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_TORSO = SceneEntityCfg("robot", body_names=("base_link",))


def torso_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _TORSO
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2:3]


def body_heights(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT):
  """World height of every robot body (the teacher's "full bodies state")."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, :, 2]


def root_quat_canonical(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT):
  """Root orientation (w, x, y, z) with w >= 0, so q and -q map to one input."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.root_link_quat_w
  return torch.where(q[:, :1] < 0, -q, q)


def feet_contact_forces(
  env: ManagerBasedRlEnv, sensor_name: str, scale: float
) -> torch.Tensor:
  """Net ground contact force on each foot (world frame), flattened, times ``scale``."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return sensor.data.force.flatten(1) * scale


def assist_wrench(
  env: ManagerBasedRlEnv, force_scale: float, torque_scale: float
) -> torch.Tensor:
  """Eq. 4 wrench for the next step, normalized by its maxima (privileged)."""
  w = get_wrench(env)
  return torch.cat((w[:, :3] * force_scale, w[:, 3:] * torque_scale), dim=-1)
