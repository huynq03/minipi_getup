"""Force-guided external assistance (FTSR paper, Eq. 4) and its constraint costs.

    [F; T] = (1 - exp(-mu * (h_cmd_i - h))) * sat01(1 - t / t_tag)
             * [F_max * n ; T_max * log(R_tag R^-1)]

- ``h`` is the torso (``base_link``) height and ``h_cmd_i`` the target height of the
  current reward stage (``stages.py``). The paper writes "CoM height". The released
  code and the stage sets use the base height, so the torso is used here too.
- The height factor is clamped at zero. Above ``h_cmd_i`` the paper's expression
  turns negative and would pull the robot down; the released code clamps it as well.
- ``t`` counts env steps (``env.common_step_counter``), and ``t_tag = end_iteration *
  steps_per_iteration``. From ``t_tag`` on, the wrench is exactly zero, which is the
  paper's ``d_i = 0`` point.
- ``log(R_tag R^-1)`` is the rotation vector, in the world frame, that would bring the
  torso upright. ``R_tag`` is the current orientation with its tilt removed
  (swing-twist split about world z). Yaw is never assisted, and the target stays
  well defined while lying, where Euler yaw is singular.
- The wrench acts on the torso CoM through MuJoCo ``xfrc_applied`` (world frame).
  Nothing is added to the joint actions.

The wrench is computed in ``mode="step"`` after rewards, so the value stored in
``env.extras`` is the wrench applied during the *next* env step. The runner reads it
before ``env.step`` and books it as that transition's constraint cost
``C = (|F| / m g, |T| / T_max) * step_dt``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# env.extras keys shared with the stage manager, observations and the runner.
WRENCH_KEY = "ftsr_assist_wrench"
COST_KEY = "ftsr_constraint_cost"
H_CMD_KEY = "ftsr_h_cmd"

_TORSO = SceneEntityCfg("robot", body_names=("base_link",))


def get_wrench(env: ManagerBasedRlEnv) -> torch.Tensor:
  wrench = env.extras.get(WRENCH_KEY)
  if wrench is None:
    return torch.zeros(env.num_envs, 6, device=env.device)
  return wrench


def uprighting_rotvec(quat_w: torch.Tensor) -> torch.Tensor:
  """World-frame rotation vector of the shortest rotation taking body z to world z.

  Equals ``log(R_tag R^-1)`` with ``R_tag`` = ``R`` minus its tilt. Its norm is the
  tilt angle (0 upright, pi/2 lying, pi upside down).
  """
  w, x, y, z = quat_w.unbind(-1)
  body_z = torch.stack(
    (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1
  )
  # body_z x world_z = (bz_y, -bz_x, 0).
  axis = torch.stack(
    (body_z[:, 1], -body_z[:, 0], torch.zeros_like(body_z[:, 0])), dim=-1
  )
  sin = axis.norm(dim=-1, keepdim=True)
  angle = torch.atan2(sin, body_z[:, 2:3])
  # Exactly upside down the axis is undefined; pick the body x axis (any horizontal
  # axis rights the robot).
  fallback = torch.zeros_like(axis)
  fallback[:, 0] = 1.0
  unit = torch.where(sin > 1e-6, axis / sin.clamp(min=1e-6), fallback)
  return unit * angle


class ftsr_assist:
  """Eq. 4 assistance on the torso. Configurable; logs the applied F and T."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _TORSO)
    asset_cfg.resolve(env.scene)
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._wrench = torch.zeros(env.num_envs, 6, device=env.device)
    self._cost = torch.zeros(env.num_envs, 2, device=env.device)
    env.extras[WRENCH_KEY] = self._wrench
    env.extras[COST_KEY] = self._cost

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._wrench[env_ids] = 0.0
    self._cost[env_ids] = 0.0
    self._write()

  def _write(self) -> None:
    self._asset.write_external_wrench_to_sim(
      self._wrench[:, None, :3], self._wrench[:, None, 3:], body_ids=self._body_ids
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    f_max: float,
    t_max: float,
    mu: float,
    end_iteration: int,
    robot_weight: float,
    steps_per_iteration: int = 24,
    default_height_cmd: float = 0.335,
    settle_steps: int = 0,
    asset_cfg: SceneEntityCfg = _TORSO,
  ) -> None:
    del env_ids  # Step mode: all envs.
    t_tag = end_iteration * steps_per_iteration
    time_coeff = min(1.0, max(0.0, 1.0 - env.common_step_counter / t_tag))
    if time_coeff <= 0.0:
      # Past t_tag: the assistance is gone for good.
      if self._wrench.any():
        self._wrench.zero_()
        self._cost.zero_()
        self._write()
    else:
      h_cmd = float(env.extras.get(H_CMD_KEY, default_height_cmd))
      height = self._asset.data.body_link_pos_w[:, self._body_ids[0], 2]
      k = 1.0 - torch.exp(-mu * torch.clamp(h_cmd - height, min=0.0))
      k = k * time_coeff
      # No assistance while the reset pose settles (actions are held then too).
      if settle_steps > 0:
        k = k * (env.episode_length_buf >= settle_steps).float()
      quat = self._asset.data.body_link_quat_w[:, self._body_ids[0]]
      self._wrench[:, :2] = 0.0
      self._wrench[:, 2] = k * f_max
      self._wrench[:, 3:] = (k * t_max).unsqueeze(-1) * uprighting_rotvec(quat)
      # Scaled by step_dt like the rewards, so cost returns stay O(1) and the cost
      # critic's loss doesn't dominate the shared gradient clip. The cost advantages
      # are standardized before Eq. 8, so the scale doesn't change A_bar.
      self._cost[:, 0] = self._wrench[:, 2].abs() / robot_weight * env.step_dt
      self._cost[:, 1] = self._wrench[:, 3:].norm(dim=-1) / t_max * env.step_dt
      self._write()

    log = env.extras.setdefault("log", {})
    force = self._wrench[:, 2]
    torque = self._wrench[:, 3:].norm(dim=-1)
    log["Assist/time_coeff"] = time_coeff
    log["Assist/force_mean"] = force.mean()
    log["Assist/force_max"] = force.max()
    log["Assist/torque_mean"] = torque.mean()
    log["Assist/torque_max"] = torque.max()
