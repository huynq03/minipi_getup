"""Isolated environment delta for arbitrary-fall HoST safe recovery."""

from __future__ import annotations

import math

import torch

from minipi_getup.host.env import LeggedRobot_Pi, torch_rand_float
from minipi_getup.host_safe_recovery.asset import REAL_TORQUE

RESET_CLASSES = ("supine", "prone", "left", "right", "intermediate")


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product for MuJoCo-order (w, x, y, z) quaternions."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    dim=-1,
  )


class SafeRecoveryEnv(LeggedRobot_Pi):
  """HoST environment with real plant limits, diverse falls, and 3 safety rewards."""

  def _create_envs(self):
    super()._create_envs()
    limits = []
    for name in self.dof_names:
      kind = next(k for k in REAL_TORQUE if name.endswith(f"_{k}_joint"))
      limits.append(REAL_TORQUE[kind])
    self.torque_limits = torch.tensor(limits, device=self.device)

  def _init_buffers(self):
    super()._init_buffers()
    self.reset_class = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.overspeed_peak_by_joint = torch.zeros_like(self.dof_vel)
    self.overspeed_steps_by_joint = torch.zeros_like(self.dof_vel)
    sigma, weight = [], []
    for name in self.dof_names:
      kind = next(k for k in self.cfg.safe_recovery.pose_sigma if name.endswith(f"_{k}_joint"))
      sigma.append(self.cfg.safe_recovery.pose_sigma[kind])
      weight.append(self.cfg.safe_recovery.pose_weight[kind])
    self.pose_sigma = torch.tensor(sigma, device=self.device).unsqueeze(0)
    self.pose_weight = torch.tensor(weight, device=self.device).unsqueeze(0)

  def _reset_dofs(self, env_ids):
    """Random valid-ROM fallen joints; velocities begin at rest."""
    n = len(env_ids)
    lower = self.robot.data.joint_pos_limits[0, :, 0]
    upper = self.robot.data.joint_pos_limits[0, :, 1]
    center = self.default_dof_pos.squeeze(0)
    half_span = 0.25 * (upper - lower)
    q = center + torch_rand_float(-1.0, 1.0, (n, self.num_dof), self.device) * half_span
    margin = 0.02
    self.dof_pos[env_ids] = torch.maximum(
      torch.minimum(q, upper.unsqueeze(0) - margin), lower.unsqueeze(0) + margin
    )
    self.dof_vel[env_ids] = 0.0
    self.robot.write_joint_state_to_sim(
      self.dof_pos[env_ids], self.dof_vel[env_ids], env_ids=env_ids
    )

  def _reset_root_states(self, env_ids):
    """Uniform mixture of four named falls plus random intermediate tilts."""
    n = len(env_ids)
    cls = torch.randint(0, len(RESET_CLASSES), (n,), device=self.device)
    self.reset_class[env_ids] = cls
    q = torch.zeros(n, 4, device=self.device)
    h = math.sqrt(0.5)
    canonical = torch.tensor(
      [[h, 0, -h, 0], [h, 0, h, 0], [h, h, 0, 0], [h, -h, 0, 0]],
      device=self.device,
    )
    for k in range(4):
      mask = cls == k
      q[mask] = canonical[k]
    mask = cls == 4
    count = int(mask.sum())
    if count:
      angle = torch_rand_float(math.pi / 3, math.pi, (count,), self.device)
      azimuth = torch_rand_float(-math.pi, math.pi, (count,), self.device)
      q[mask, 0] = torch.cos(0.5 * angle)
      q[mask, 1] = torch.sin(0.5 * angle) * torch.cos(azimuth)
      q[mask, 2] = torch.sin(0.5 * angle) * torch.sin(azimuth)
    yaw = torch_rand_float(-math.pi, math.pi, (n,), self.device)
    qyaw = torch.zeros_like(q)
    qyaw[:, 0], qyaw[:, 3] = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    q = _quat_mul(qyaw, q)
    q /= torch.linalg.vector_norm(q, dim=1, keepdim=True)

    self.root_states[env_ids] = self.base_init_state
    self.root_states[env_ids, :3] = self.env_origins[env_ids]
    self.root_states[env_ids, :2] += torch_rand_float(-0.03, 0.03, (n, 2), self.device)
    self.root_states[env_ids, 2] = torch_rand_float(0.34, 0.37, (n,), self.device)
    self.root_states[env_ids, 3:7] = q
    self.root_states[env_ids, 7:10] = torch_rand_float(-0.10, 0.10, (n, 3), self.device)
    self.root_states[env_ids, 10:13] = torch_rand_float(-0.20, 0.20, (n, 3), self.device)
    self.base_quat[env_ids] = q
    self.robot.write_root_state_to_sim(self.root_states[env_ids], env_ids=env_ids)

  def _standing_gate(self) -> torch.Tensor:
    cfg = self.cfg.safe_recovery
    height = torch.sigmoid(
      (self.root_states[:, 2] - cfg.gate_height_center) / cfg.gate_height_width
    )
    upright = torch.sigmoid(
      ((-self.projected_gravity[:, 2]) - cfg.gate_upright_center)
      / cfg.gate_upright_width
    )
    return height * upright

  def _reward_joint_overspeed(self):
    excess = torch.relu(torch.abs(self.dof_vel) - self.cfg.safe_recovery.overspeed_soft_rad_s)
    return torch.mean(excess.square(), dim=1)

  def _reward_default_posture(self):
    normalized = (self.dof_pos - self.default_dof_pos) / self.pose_sigma
    return self._standing_gate() * torch.mean(self.pose_weight * normalized.square(), dim=1)

  def _reward_stable_standing(self):
    cfg = self.cfg.safe_recovery
    motion = cfg.stable_joint_velocity_weight * torch.mean(self.dof_vel.square(), dim=1)
    motion += cfg.stable_base_angular_velocity_weight * torch.sum(
      self.base_ang_vel.square(), dim=1
    )
    return self._standing_gate() * motion

  def _substep_monitor(self):
    super()._substep_monitor()
    speed = torch.abs(self.dof_vel)
    self.overspeed_peak_by_joint = torch.maximum(self.overspeed_peak_by_joint, speed)
    self.overspeed_steps_by_joint += (
      speed > self.cfg.safe_recovery.overspeed_soft_rad_s
    )

  def _log_reset_metrics(self, env_ids):
    super()._log_reset_metrics(env_ids)
    peaks = self.overspeed_peak_by_joint[env_ids].max(0).values
    denom = (
      self.episode_length_buf[env_ids].float().sum()
      * self.cfg.control.decimation
    ).clamp_min(1.0)
    fractions = self.overspeed_steps_by_joint[env_ids].sum(0) / denom
    for i, name in enumerate(self.dof_names):
      self.reset_stats[f"overspeed_peak_{name}"] = peaks[i]
      self.reset_stats[f"overspeed_fraction_{name}"] = fractions[i]
    self.reset_stats["velocity_ceiling_violation"] = (
      peaks > self.cfg.safe_recovery.evaluation_ceiling_rad_s
    ).float().mean()

  def reset_idx(self, env_ids):
    super().reset_idx(env_ids)
    if len(env_ids):
      self.overspeed_peak_by_joint[env_ids] = 0
      self.overspeed_steps_by_joint[env_ids] = 0
