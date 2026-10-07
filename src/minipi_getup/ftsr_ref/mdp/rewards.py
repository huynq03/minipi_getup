"""Release reward terms (getup_gym ``common/reward_functions.py``), stage-weighted.

Every term is the release's raw formula with only robot-specific replacements; the
weight of the current stage (``stages.py``) multiplies it inside the term, so the
reward manager's own weight is 1. mjlab then multiplies by ``step_dt`` = 0.02, exactly
the release's ``weight * raw * dt``. The per-term table (release name, formula,
g2/g3/g4 weights, paper Table II, Mini-Pi form, reason) is in
``config/rewards.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

from minipi_getup.ftsr_ref.config.robot import JOINT_SUFFIXES
from minipi_getup.ftsr_ref.mdp.stages import get_stage_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _stage_weight(env: ManagerBasedRlEnv, stage_weights: tuple[float, ...]) -> float:
  return stage_weights[get_stage_state(env).update(env)]


def _act(env):
  return env.action_manager.get_term("joint_pos")


def _robot(env):
  return env.scene["robot"]


def _base_height(env) -> torch.Tensor:
  robot = _robot(env)
  return robot.data.body_link_pos_w[:, robot.find_bodies(("base_link",))[0][0], 2]


def _q(env):
  return _robot(env).data.joint_pos[:, _act(env)._ids]


def _qd(env):
  return _robot(env).data.joint_vel[:, _act(env)._ids]


def _tau(env):
  return _robot(env).data.qfrc_actuator[:, _act(env)._ids]


def _feet_fz(env, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return sensor.data.force[..., 2].abs()  # (N, 2)


class StagedTerm:
  """Reward term = stage weight x raw term; zero (and not evaluated) at weight 0."""

  raw = None  # set by subclasses / staged()

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.env = env

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env, stage_weights, **kwargs) -> torch.Tensor:
    w = _stage_weight(env, stage_weights)
    if w == 0.0:
      return torch.zeros(env.num_envs, device=env.device)
    return w * self.compute(env, **kwargs)

  def compute(self, env, **kwargs) -> torch.Tensor:
    raise NotImplementedError


def staged(fn):
  """Stateless raw term -> stage-weighted term class."""

  class _T(StagedTerm):
    def compute(self, env, **kwargs):
      return fn(env, **kwargs)

  _T.__name__ = fn.__name__
  _T.__doc__ = fn.__doc__
  return _T


# Raw terms (release names). -------------------------------------------------------


@staged
def pen_termination(env) -> torch.Tensor:
  """reset & ~time_out."""
  return env.termination_manager.terminated.float()


@staged
def pen_base_orientation_l2(env) -> torch.Tensor:
  """0.02 |g_xy|^2 + 2 (g_z + 1)^2 - clamp(exp(-|g_xy|^2 / 0.25), 0, 0.1)."""
  g = _robot(env).data.projected_gravity_b
  xy = torch.sum(torch.square(g[:, :2]), dim=1)
  z = torch.square(g[:, 2] + 1.0)
  return 0.02 * xy + 2.0 * z - torch.exp(-xy / 0.25).clamp(0.0, 0.1)


@staged
def pen_base_orientation_z_l2(env) -> torch.Tensor:
  """1[-g_z < -0.2] (upside down)."""
  g = _robot(env).data.projected_gravity_b
  return (-g[:, 2] < -0.2).float()


@staged
def track_base_height_exp(env, sigma: float) -> torch.Tensor:
  """exp(-|h - h_t| / sigma), h_t = height-reward target of the current stage."""
  h_t = get_stage_state(env).h_reward
  return torch.exp(-torch.abs(_base_height(env) - h_t) / sigma)


class pen_dof_acc_l2(StagedTerm):
  """sum(((qdot_{t-1} - qdot_t) / dt)^2), dt = policy step (release ``self.dt``)."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._prev = torch.zeros(env.num_envs, _act(env).action_dim, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def compute(self, env) -> torch.Tensor:
    qd = _qd(env)
    acc = (self._prev - qd) / env.step_dt
    acc = torch.where(self._fresh.unsqueeze(-1), torch.zeros_like(acc), acc)
    self._prev = qd.clone()
    self._fresh[:] = False
    return torch.sum(torch.square(acc), dim=1)

  def __call__(self, env, stage_weights) -> torch.Tensor:
    # Always update the previous velocity, whatever the weight.
    w = _stage_weight(env, stage_weights)
    out = self.compute(env)
    return w * out if w != 0.0 else torch.zeros_like(out)


@staged
def pen_action_rate_l2(env) -> torch.Tensor:
  """sum((a_{t-1} - a_t)^2), raw-clipped actions."""
  a = _act(env)
  return torch.sum(torch.square(a.prev_raw_action - a.raw_action), dim=1)


@staged
def pen_action_smoothness_l2(env) -> torch.Tensor:
  """sum((a_t - 2 a_{t-1} + a_{t-2})^2)."""
  a = _act(env)
  d = a.raw_action - 2.0 * a.prev_raw_action + a.prev_prev_raw_action
  return torch.sum(torch.square(d), dim=1)


def _both_feet_contact(env, sensor_name: str, threshold: float) -> torch.Tensor:
  return ((_feet_fz(env, sensor_name) > threshold).sum(-1) > 1).float()


@staged
def pen_dof_pos_bias_l2(env, sensor_name: str, contact_threshold: float):
  """(sum_j [(q_L,j - d_j)^2 + |q_R,j - d_j|])^2 * 1[both feet in contact].

  Release: j = hip roll, femur pitch, tibia pitch (all non-wheel leg joints), square
  on the left leg and absolute value on the right (quirk Q10, ported literally).
  Mini-Pi: all six leg joints.
  """
  a = _act(env)
  q = _q(env) - a.default
  n = len(JOINT_SUFFIXES)
  right, left = q[:, :n], q[:, n:]
  s = (torch.square(left) + torch.abs(right)).sum(-1)
  return torch.square(s) * _both_feet_contact(env, sensor_name, contact_threshold)


@staged
def pen_two_leg_bias_l2(env) -> torch.Tensor:
  """sum_j (q_L,j - q_R,j)^2, raw difference (quirk Q9: JiaRan's roll axes, like
  Mini-Pi's, point the same way on both legs, so the literal port keeps its meaning)."""
  q = _q(env)
  n = len(JOINT_SUFFIXES)
  return torch.sum(torch.square(q[:, n:] - q[:, :n]), dim=1)


@staged
def pen_no_fly_l2(env, sensor_name: str, contact_threshold: float) -> torch.Tensor:
  """1[#feet with F_z > threshold != 2]."""
  return ((_feet_fz(env, sensor_name) > contact_threshold).sum(-1) != 2).float()


@staged
def rew_wheel_contact_force(env, sensor_name: str, weight: float, width: float):
  """exp(-|sum F_z,feet - m g| / width) (release: wheels, 0.05/N, 10 M)."""
  fz = _feet_fz(env, sensor_name).sum(-1)
  return torch.exp(-torch.abs(fz - weight) / width)


@staged
def pen_torques_l2(env) -> torch.Tensor:
  """sum(tau^2)."""
  return torch.sum(torch.square(_tau(env)), dim=1)


@staged
def track_lin_vel_xy_exp(env, sigma: float, min_height: float) -> torch.Tensor:
  """exp(-|v_xy - v_cmd|^2 / sigma) * 1[h > min_height]."""
  cmd = env.command_manager.get_command("twist")
  v = _robot(env).data.root_link_lin_vel_b[:, :2]
  err = torch.sum(torch.square(cmd[:, :2] - v), dim=1)
  return torch.exp(-err / sigma) * (_base_height(env) > min_height).float()


@staged
def track_ang_vel_yaw_exp(env, sigma: float) -> torch.Tensor:
  """exp(-(w_z - w_cmd)^2 / sigma)."""
  cmd = env.command_manager.get_command("twist")
  w = _robot(env).data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(cmd[:, 2] - w) / sigma)


@staged
def pen_dof_vel_l2(env) -> torch.Tensor:
  """sum(qdot^2), leg joints."""
  return torch.sum(torch.square(_qd(env)), dim=1)


@staged
def pen_feet_distance_l2(env, lo: float, hi: float) -> torch.Tensor:
  """clip(lo - d, 0, 1) + clip(d - hi, 0, 1), d = xy distance of the feet."""
  robot = _robot(env)
  ids = robot.find_bodies(
    ("r_ankle_roll_link", "l_ankle_roll_link"), preserve_order=True
  )[0]
  p = robot.data.body_link_pos_w[:, ids, :2]
  d = torch.norm(p[:, 0] - p[:, 1], dim=-1)
  return torch.clamp(lo - d, 0.0, 1.0) + torch.clamp(d - hi, 0.0, 1.0)


@staged
def pen_max_velocity_l2(env) -> torch.Tensor:
  """+1 if v_x in [0.8, 1.1] x cmd_x (sign-aware, release form), -1 otherwise."""
  v = _robot(env).data.root_link_lin_vel_b[:, 0]
  c = env.command_manager.get_command("twist")[:, 0]
  vel_low_1, vel_high_1 = v > c * 0.8, v < c * 1.1
  vel_low, vel_high = v < c * 0.8, v > c * 1.1
  c_low, c_high = c < 0.0, c >= 0.0
  desired = (~(vel_low | vel_high)) & c_high
  desired_1 = ~(vel_low_1 | vel_high_1) & c_low
  # Same assignment order as the release (later writes win), without masked
  # indexing (a host sync).
  r = torch.zeros_like(v)
  r = torch.where(vel_low | vel_low_1, -1.0, r)
  r = torch.where(vel_high | vel_high_1, -1.0, r)
  return torch.where(desired | desired_1, 1.0, r)


@staged
def pen_dof_pos_limits(env, soft: float) -> torch.Tensor:
  """sum of violations beyond the soft limits (mid +- soft x half range)."""
  a = _act(env)
  mid = 0.5 * (a.q_min + a.q_max)
  half = 0.5 * (a.q_max - a.q_min) * soft
  q = _q(env)
  out = -(q - (mid - half)).clamp(max=0.0) + (q - (mid + half)).clamp(min=0.0)
  return out.sum(dim=1)


@staged
def pen_torque_limits(env, limit: float, soft: float) -> torch.Tensor:
  """sum(relu(|tau| - soft x limit))."""
  return (torch.abs(_tau(env)) - limit * soft).clamp(min=0.0).sum(dim=1)


@staged
def pen_joint_power_l2(env, max_height: float) -> torch.Tensor:
  """sum(|qdot| |tau|) * 1[h < max_height]."""
  p = torch.sum(torch.abs(_qd(env)) * torch.abs(_tau(env)), dim=1)
  return p * (_base_height(env) < max_height).float()


@staged
def qd_soft_envelope(env, limit: float) -> torch.Tensor:
  """sum(relu(|qdot| - limit)^2): operational soft joint-speed envelope.

  Not a release term: the user's operational contract (HARDWARE_CONSTRAINT), a
  moderate penalty that leaves sub-limit motion free.
  """
  return torch.square(torch.relu(_qd(env).abs() - limit)).sum(dim=1)
