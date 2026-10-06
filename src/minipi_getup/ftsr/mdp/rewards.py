"""FTSR reward terms (paper Table II) adapted to the legged Mini-Pi.

Every term returns a non-negative value; the stage weights carry the sign. mjlab
multiplies each term by ``weight * step_dt`` (0.02 s), exactly as getup_gym's
``RewardManager`` does (``weight * reward * dt``), so Table II weights carry over.
Exponential kernels whose width is a length are rescaled to Mini-Pi's size (see
``config/stage_rewards.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from minipi_getup.ftsr.mdp.assistance import H_CMD_KEY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_TORSO = SceneEntityCfg("robot", body_names=("base_link",))


def _torso_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]


def stage_height_exp(
  env: ManagerBasedRlEnv,
  coef: float,
  default_height_cmd: float = 0.335,
  asset_cfg: SceneEntityCfg = _TORSO,
) -> torch.Tensor:
  """Base heig.: exp(-coef (h - h_cmd_i)^2), h_cmd_i from the current stage."""
  h_cmd = float(env.extras.get(H_CMD_KEY, default_height_cmd))
  h = _torso_height(env, asset_cfg)
  return torch.exp(-coef * torch.square(h - h_cmd))


def _upright_gate(
  env: ManagerBasedRlEnv, min_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  return (_torso_height(env, asset_cfg) > min_height).float()


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  coef: float,
  command_name: str,
  min_height: float,
  asset_cfg: SceneEntityCfg = _TORSO,
) -> torch.Tensor:
  """Lin. vel.: exp(-coef |v_xy - v_cmd|^2), only once the torso is up.

  Gated like the released code (``base_height > 0.6`` of a 0.75 m stance), so a robot
  still on the ground earns no tracking reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  err = torch.sum(torch.square(cmd[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), 1)
  return torch.exp(-coef * err) * _upright_gate(env, min_height, asset_cfg)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  coef: float,
  command_name: str,
  min_height: float,
  asset_cfg: SceneEntityCfg = _TORSO,
) -> torch.Tensor:
  """Ang. vel.: exp(-coef (w_z - w_cmd)^2), only once the torso is up."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  err = torch.square(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-coef * err) * _upright_gate(env, min_height, asset_cfg)


def orientation_xy_norm(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Orient.: |g_xy|_2 (projected gravity in the body frame)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)


def upside_down(
  env: ManagerBasedRlEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """1 while the torso points down (released ``pen_base_orientation_z_l2``).

  |g_xy| is zero both upright and upside down, so Table II's orientation term alone
  can't tell them apart.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return (asset.data.projected_gravity_b[:, 2] > threshold).float()


def joint_deviation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Dof pos.: |q - q_nominal|^2. Mini-Pi's nominal stance is all joints at 0."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(q - q0), dim=-1)


def joint_power_sq(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Dof ener.: |qdot * tau|^2 (per-joint mechanical power, squared and summed)."""
  asset: Entity = env.scene[asset_cfg.name]
  power = asset.data.joint_vel * asset.data.qfrc_actuator
  return torch.sum(torch.square(power), dim=-1)


def torque_limit_excess(
  env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Sum over joints of |tau| above ``limit`` (released ``pen_torque_limits``)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.relu(asset.data.actuator_force.abs() - limit).sum(dim=-1)


def leg_mirror_error(
  env: ManagerBasedRlEnv,
  same_sign: tuple[str, ...] = ("hip_pitch_joint", "calf_joint", "ankle_pitch_joint"),
  mirrored: tuple[str, ...] = ("hip_roll_joint", "thigh_joint", "ankle_roll_joint"),
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Leg bias: |q_left - q_right|_2 with the roll/yaw axes mirrored.

  All Mini-Pi joint axes point the same way on both legs, so pitch joints mirror as
  q_l = q_r and roll / yaw joints as q_l = -q_r.
  """
  asset: Entity = env.scene[asset_cfg.name]
  names = asset.joint_names
  q = asset.data.joint_pos
  diffs = [q[:, names.index("l_" + s)] - q[:, names.index("r_" + s)] for s in same_sign]
  diffs += [q[:, names.index("l_" + s)] + q[:, names.index("r_" + s)] for s in mirrored]
  return torch.stack(diffs, dim=-1).norm(dim=-1)


def no_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """No fly: 1 when neither foot touches the ground."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found.sum(dim=-1) == 0).float()


def feet_support_force(
  env: ManagerBasedRlEnv, sensor_name: str, robot_weight: float, width: float
) -> torch.Tensor:
  """Leg usage (paper "Wheel force"): exp(-|sum F_z,feet - m g| / width).

  The released code uses exp(-0.05 |F - M g|) with forces in N. Here the width is a
  fraction of Mini-Pi's weight, so the kernel has the same meaning on a lighter robot.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  fz = sensor.data.force[..., 2].abs().sum(dim=-1)
  return torch.exp(-torch.abs(fz - robot_weight) / width)


def feet_lateral_distance(
  env: ManagerBasedRlEnv, target: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Feet dist.: | |y_l - y_r| (base frame) - target |, target = the stance width."""
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [B, 2, 3]
  rel = feet - asset.data.root_link_pos_w[:, None, :]
  # Rotate into the heading frame (yaw only), so torso pitch doesn't matter.
  yaw = asset.data.heading_w
  y_b = -torch.sin(yaw)[:, None] * rel[..., 0] + torch.cos(yaw)[:, None] * rel[..., 1]
  return torch.abs(torch.abs(y_b[:, 0] - y_b[:, 1]) - target)


class joint_acc_fd_l2:
  """DOF acc.: sum((qdot_t - qdot_{t-1}) / step_dt)^2, as in getup_gym.

  The finite difference over one policy step matches the reference. MuJoCo's
  instantaneous ``qacc`` at 2 ms spikes on every contact impact, and with Table II's
  weights it would dominate the reward.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[_ROBOT.name]
    self._prev = asset.data.joint_vel.clone()
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def __call__(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel
    acc = (qd - self._prev) / env.step_dt
    acc = torch.where(self._fresh.unsqueeze(-1), torch.zeros_like(acc), acc)
    self._prev = qd.clone()
    self._fresh[:] = False
    return torch.sum(torch.square(acc), dim=-1)


def is_standing(
  env: ManagerBasedRlEnv,
  min_height: float = 0.31,
  min_up: float = 0.95,
  asset_cfg: SceneEntityCfg = _TORSO,
) -> torch.Tensor:
  """1 while the torso is above 0.9 x stance height and within ~18 deg of upright."""
  asset: Entity = env.scene[asset_cfg.name]
  up = -asset.data.projected_gravity_b[:, 2]
  return ((_torso_height(env, asset_cfg) > min_height) & (up > min_up)).float()


def fell_over(
  env: ManagerBasedRlEnv,
  min_height: float,
  max_tilt: float,
  asset_cfg: SceneEntityCfg = _TORSO,
) -> torch.Tensor:
  """Torso below ``min_height`` or tilted more than ``max_tilt`` rad (walk pretrain)."""
  asset: Entity = env.scene[asset_cfg.name]
  up = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
  return (_torso_height(env, asset_cfg) < min_height) | (torch.acos(up) > max_tilt)


def feet_gait(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period: float,
  offset: tuple[float, ...],
  stance_fraction: float,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Share of feet whose contact state matches the gait clock (walking stage only).

  Foot ``j`` should be in stance while ``(t / period + offset_j) % 1 < stance_fraction``
  and in swing otherwise. Zero for stand commands. Same clock as ``gait_phase``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  contact = sensor.data.current_contact_time > 0
  t = (env.episode_length_buf * env.step_dt / period).unsqueeze(-1)
  offs = torch.as_tensor(offset, device=env.device, dtype=t.dtype)
  stance = (t + offs) % 1.0 < stance_fraction
  reward = (stance == contact).float().mean(dim=-1)
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  return reward * (torch.linalg.norm(cmd, dim=-1) > command_threshold).float()
