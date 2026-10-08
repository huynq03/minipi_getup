"""Force-guided external assistance: paper Eq. 4 (PAPER_COMPLETION).

    [F; T] = (1 - exp(-mu (h_cmd_i - h))) sat_[0,1](1 - t / t_tag)
             x [F_max n ; T_max log(R_tag R^-1)]

Why Eq. 4 and not the release's ``_base_force_pull_up`` (FTSR_REFERENCE_AUDIT_V2.md
Sec. 4, 8): the release uses a linear height factor toward a fixed 0.75 m and a
world-x torque built from the quaternion's y component, which only works for its
single reset orientation. F and T are also the CPO constraint costs (Eq. 5-8), which
the release never wires up. One definition only: nothing else applies forces.

- ``h``: base_link height above the ground (the release uses the root height too).
- ``h_cmd_i``: target height of the current reward stage (``stages.py``).
- Height factor clamped at 0 above ``h_cmd_i`` (the release clamps its force at 0).
- Time factor: ``t`` = env steps (``common_step_counter``), ``t_tag`` =
  ``end_iteration * steps_per_iteration``. From ``t_tag`` on the wrench is exactly zero
  (``d_i = 0``); ``time_coeff`` is asserted to be zero there.
- ``n``: world +z. ``log(R_tag R^-1)``: world-frame rotation vector of the shortest
  rotation bringing body z to world z, i.e. ``R_tag`` = R with its tilt removed (yaw is
  never assisted; well defined while lying, where Euler yaw is singular).
- Applied to the base_link CoM every physics step (``xfrc_applied``), like the release
  (every physics step), and zero during the passive window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AssistCfg:
  f_max: float
  """N. Default: the release's 400 N / (27.669 kg g) = 1.474 m g, times Mini-Pi m g."""
  t_max: float
  """N m / rad (UNRESOLVED calibration, see env_cfg)."""
  mu: float
  """1/m (UNRESOLVED calibration, see env_cfg)."""
  end_iteration: int = 3000
  steps_per_iteration: int = 24

  @property
  def t_tag(self) -> int:
    return self.end_iteration * self.steps_per_iteration


def time_coeff(step: int, cfg: AssistCfg) -> float:
  """sat_[0,1](1 - t / t_tag); exactly 0.0 for t >= t_tag."""
  if step >= cfg.t_tag:
    return 0.0
  return min(1.0, max(0.0, 1.0 - step / cfg.t_tag))


def uprighting_rotvec(quat_w: torch.Tensor) -> torch.Tensor:
  """World-frame rotation vector taking body z to world z (norm = tilt angle).

  quat_w: (..., 4) as (w, x, y, z).
  """
  w, x, y, z = quat_w.unbind(-1)
  body_z = torch.stack(
    (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1
  )
  # body_z x world_z = (bz_y, -bz_x, 0).
  axis = torch.stack(
    (body_z[..., 1], -body_z[..., 0], torch.zeros_like(body_z[..., 0])), dim=-1
  )
  sin = axis.norm(dim=-1, keepdim=True)
  angle = torch.atan2(sin, body_z[..., 2:3])
  # Exactly upside down the axis is undefined; any horizontal axis rights the robot.
  fallback = torch.zeros_like(axis)
  fallback[..., 0] = 1.0
  unit = torch.where(sin > 1e-6, axis / sin.clamp(min=1e-6), fallback)
  return unit * angle


def eq4_wrench(
  height: torch.Tensor,
  quat_w: torch.Tensor,
  h_cmd: float,
  t_coeff: float,
  cfg: AssistCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """World-frame force (N, 3) and torque (N, 3) of Eq. 4."""
  k = (1.0 - torch.exp(-cfg.mu * torch.clamp(h_cmd - height, min=0.0))) * t_coeff
  force = torch.zeros(height.shape[0], 3, device=height.device)
  force[:, 2] = k * cfg.f_max
  torque = (k * cfg.t_max).unsqueeze(-1) * uprighting_rotvec(quat_w)
  return force, torque
