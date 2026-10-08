"""Terminations. Recovery: time-out only (release, quirk Q8). Walking init: falls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def fell_over(
  env: ManagerBasedRlEnv, min_height: float, max_tilt: float
) -> torch.Tensor:
  """Base below ``min_height`` or tilted more than ``max_tilt`` rad."""
  robot = env.scene["robot"]
  body = robot.find_bodies(("base_link",))[0][0]
  h = robot.data.body_link_pos_w[:, body, 2]
  up = (-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
  return (h < min_height) | (torch.acos(up) > max_tilt)
