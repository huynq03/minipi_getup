"""Reset distribution (paper Sec. III-A, Table III "Initial base euler").

Four canonical fallen orientations (supine, prone, left side, right side) with
U(-0.3, 0.3) rad roll/pitch noise and uniform yaw (paper). The release resets to one
orientation only (quirk Q7, PAPER_COMPLETION).

Joints: U(-0.3, 0.3) rad added to the nominal stance (all zeros), clamped to the
physical ranges. Paper / release scale the default angles by U(0.5, 1.5) / U(0.1, 1.5),
which is a no-op for Mini-Pi's zero defaults; the additive spread matches the release's
spread on its nonzero joints (femur 0.4 x [0.1, 1.5]) (ROBOT_ADAPTATION).

Fallen poses start ``drop`` m above the ground at rest and settle under gravity in the
passive window (deploy Passive: kp 0, kd 1); actions and assistance are off until it
ends, so the policy never exploits the drop. The settled distribution is measured by
validate.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


class ResetPose(TypedDict):
  roll: float
  pitch: float
  height: float


# Body x forward, y left, z up. Supine: -pi/2 about y (front up); prone: +pi/2.
# Left side: -pi/2 about x (body y down); right side: +pi/2.
FALLEN_POSES: dict[str, ResetPose] = {
  "supine": {"roll": 0.0, "pitch": -math.pi / 2, "height": 0.16},
  "prone": {"roll": 0.0, "pitch": math.pi / 2, "height": 0.16},
  "left_side": {"roll": -math.pi / 2, "pitch": 0.0, "height": 0.16},
  "right_side": {"roll": math.pi / 2, "pitch": 0.0, "height": 0.16},
}
FALLEN_POSE_NAMES = tuple(FALLEN_POSES)
STANDING_POSE: ResetPose = {"roll": 0.0, "pitch": 0.0, "height": 0.352}

POSE_KEY = "ftsr_reset_pose"


def reset_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  poses: tuple[ResetPose, ...],
  euler_noise: float,
  joint_noise: float,
  by_env_index: bool = False,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Reset each env to one of ``poses`` (uniform, or ``env_id % len`` if
  ``by_env_index``), perturbed, at rest. The pose index is kept in
  ``env.extras[POSE_KEY]`` for per-pose statistics."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  n = len(env_ids)
  dev = env.device
  asset = env.scene[asset_cfg.name]
  if by_env_index:
    choice = env_ids.long() % len(poses)
  else:
    choice = torch.randint(0, len(poses), (n,), device=dev)
  table = torch.tensor(
    [[p["roll"], p["pitch"], p["height"]] for p in poses], device=dev
  )
  chosen = table[choice]

  roll = chosen[:, 0] + sample_uniform(-euler_noise, euler_noise, (n,), dev)
  pitch = chosen[:, 1] + sample_uniform(-euler_noise, euler_noise, (n,), dev)
  yaw = sample_uniform(-math.pi, math.pi, (n,), dev)
  quat = quat_from_euler_xyz(roll, pitch, yaw)
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, 2] += chosen[:, 2]
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

  limits = asset.data.soft_joint_pos_limits
  limits = limits.expand(env.num_envs, -1, -1)[env_ids]
  default = asset.data.default_joint_pos[env_ids]
  q = default + sample_uniform(-joint_noise, joint_noise, default.shape, dev)
  q = torch.clamp(q, limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

  if POSE_KEY not in env.extras:
    env.extras[POSE_KEY] = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
  env.extras[POSE_KEY][env_ids] = choice
