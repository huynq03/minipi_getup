"""Reset distribution for FTSR (paper Sec. III-A, Table III "Initial base euler")."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


class ResetPose(TypedDict):
  roll: float
  pitch: float
  height: float
  settle: bool


# Canonical base orientations. Mini-Pi's body x is forward, y left, z up.
# Supine: rotated -pi/2 about y, so the front faces up. Prone: +pi/2, front down.
# Left side: -pi/2 about x (body y points down); right side: +pi/2.
# Dropped from 0.16 m (lying the robot is ~0.08 m tall), so random joint angles don't
# start in the ground; the settle window absorbs the short fall.
FALLEN_POSES: dict[str, ResetPose] = {
  "supine": {"roll": 0.0, "pitch": -math.pi / 2, "height": 0.16, "settle": True},
  "prone": {"roll": 0.0, "pitch": math.pi / 2, "height": 0.16, "settle": True},
  "left_side": {"roll": -math.pi / 2, "pitch": 0.0, "height": 0.16, "settle": True},
  "right_side": {"roll": math.pi / 2, "pitch": 0.0, "height": 0.16, "settle": True},
}
STANDING_POSE: ResetPose = {"roll": 0.0, "pitch": 0.0, "height": 0.352, "settle": False}


def reset_pose_distribution(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  poses: tuple[ResetPose, ...],
  probabilities: tuple[float, ...],
  euler_noise: float = 0.3,
  joint_noise: float = 0.3,
  random_yaw: bool = True,
  by_env_index: bool = False,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Reset each env to one canonical pose, perturbed, at rest.

  Roll and pitch get U(-euler_noise, euler_noise) (paper: 0.3 rad), yaw is uniform.
  Joints get U(-joint_noise, joint_noise) around the nominal stance (all zeros),
  clamped to the joint limits. The paper scales the default angles by U(0.5, 1.5),
  which does nothing when the defaults are 0. Envs whose pose has ``settle`` are
  marked in ``env.extras["settle_mask"]``, so their actions (and the assist) are held
  for the settle window. ``by_env_index`` assigns pose ``env_id % len(poses)``
  instead of sampling (deterministic evaluation).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  dev = env.device

  probs = torch.tensor(probabilities, device=dev, dtype=torch.float)
  if by_env_index:
    choice = env_ids.long() % len(poses)
  else:
    choice = torch.multinomial(probs / probs.sum(), n, replacement=True)
  table = torch.tensor(
    [[p["roll"], p["pitch"], p["height"], float(p["settle"])] for p in poses],
    device=dev,
  )
  chosen = table[choice]

  roll = chosen[:, 0] + sample_uniform(-euler_noise, euler_noise, (n,), dev)
  pitch = chosen[:, 1] + sample_uniform(-euler_noise, euler_noise, (n,), dev)
  yaw = (
    sample_uniform(-math.pi, math.pi, (n,), dev)
    if random_yaw
    else torch.zeros(n, device=dev)
  )
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, 2] += chosen[:, 2]
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

  limits = asset.data.soft_joint_pos_limits
  assert limits is not None
  limits = limits.expand(env.num_envs, -1, -1)[env_ids]
  default = asset.data.default_joint_pos
  assert default is not None
  q = default[env_ids] + sample_uniform(
    -joint_noise, joint_noise, default[env_ids].shape, dev
  )
  q = torch.clamp(q, limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

  if "settle_mask" not in env.extras:
    env.extras["settle_mask"] = torch.zeros(env.num_envs, device=dev, dtype=torch.bool)
  env.extras["settle_mask"][env_ids] = chosen[:, 3] > 0.5
