"""Deployment-equivalent joint-position action with the Mini-Pi plant.

Per policy step (``process_actions``), identical to the deploy runtime contract
(docs/MINIPI_DEPLOYMENT_CONTRACT_V2.md, consequence 4)::

    a_c    = clip(a, -raw_clip, raw_clip)                 raw clip
    q_cmd  = q_default + scale * a_c                      absolute target
    q_cmd  = clip(q_cmd, q_min, q_max)                    physical per-joint clip
    q*_t   = clip(q_cmd, q*_{t-1} - s, q*_{t-1} + s)      slew, s = 3 rad/s * 0.02 s
    PD     tau = kp (q*_t - q) - kd qdot, held 10 x 2 ms  native position actuator

- ``q*_{t-1}`` is the previous commanded target. At state entry (the first actuated
  step after a reset) it is the measured pose clipped to the physical ranges.
- ``last_action`` (observation) is the raw-clipped network output ``a_c``, before the
  slew, zero at reset and during the passive window (deploy semantics).
- Passive window (``passive_steps`` after a reset, recovery task only): the deploy
  Passive state, kp 0 / kd 1 on every joint; actions, observations and assistance off.
  It replaces the release's "action 0 = hold default pose" window (classified
  HARDWARE_CONSTRAINT: the robot enters GetUp from Passive and the slew contract
  forbids jumping to the default pose).

Per physics step (``apply_actions``):

- Motor envelope: the four-quadrant DC curve of ``config/robot.py``, applied as the
  active affine piece of the actuator law (``_write_actuator_law``; implicit in the
  velocity slope). Measured qdot is never clamped.
- Eq. 4 assistance (``assistance.py``), if configured.
- Substep monitor: torque, qdot and tracking-error histograms per joint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields

from minipi_getup.ftsr_ref.config.robot import (
  JOINT_NAMES,
  JOINT_RANGES,
  PASSIVE_KD,
  MotorEnvelopeCfg,
)
from minipi_getup.ftsr_ref.mdp.assistance import AssistCfg, eq4_wrench, time_coeff
from minipi_getup.ftsr_ref.mdp.stages import STAGE_KEY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

COST_KEY = "ftsr_constraint_cost"


def motor_envelope(
  qd: torch.Tensor, motor: MotorEnvelopeCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """(tau_min, tau_max) of the linear four-quadrant DC curve (= mjlab dc_motor_clip)."""
  stall, w0, cap = motor.tau_stall, motor.omega0, motor.tau_cap
  vel_at_cap = w0 * (1.0 + cap / stall)
  v = torch.clamp(qd, -vel_at_cap, vel_at_cap)
  tau_max = torch.clamp(stall * (1.0 - v / w0), max=cap)
  tau_min = torch.clamp(stall * (-1.0 - v / w0), min=-cap)
  return tau_min, tau_max


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def plant_setup(env: ManagerBasedRlEnv, env_ids) -> None:
  """Startup event: only makes the actuator fields per-env (written by the action)."""
  del env, env_ids


class SubstepMonitor:
  """Physics-step actuator / speed statistics of actuated envs, for training logs.

  Cheap sufficient statistics only (per joint): sample count, sum and max of |tau|,
  |qdot| and |q* - q|, and threshold counts (|qdot| > 3 and > 4 rad/s, |tau| above the
  candidate rated torque, |tau| >= 0.95 cap). Everything stays on the device during
  the rollout; ``summary()`` converts once per iteration. Quantiles (p95 / p99) are
  not computed here: ``evaluate.py`` reports them from full histograms.
  """

  def __init__(self, nj: int, num_envs: int, motor: MotorEnvelopeCfg, device):
    self.motor = motor
    self.device = device
    self.nj = nj
    self.ep_qd_max = torch.zeros(num_envs, device=device)
    self.clear()

  def clear(self) -> None:
    def z(*shape):
      return torch.zeros(shape, device=self.device)

    nj = self.nj
    self.n = z()
    self.tau_sum, self.tau_max = z(nj), z(nj)
    self.qd_sum, self.qd_max = z(nj), z(nj)
    self.err_sum, self.err_max = z(nj), z(nj)
    self.qd_over3, self.qd_over4 = z(nj), z(nj)
    self.tau_over_rated, self.tau_near_cap = z(nj), z(nj)
    self.slew_sat, self.slew_n = z(), z()
    self.eps, self.eps_over4 = z(), z()

  def add(self, tau, qd, err, mask_f) -> None:
    """mask_f: (B, 1) float, 1 for actuated envs."""
    at = tau.abs() * mask_f
    aq = qd.abs() * mask_f
    ae = err.abs() * mask_f
    self.n += mask_f.sum()
    self.tau_sum += at.sum(0)
    self.qd_sum += aq.sum(0)
    self.err_sum += ae.sum(0)
    self.tau_max = torch.maximum(self.tau_max, at.amax(0))
    self.qd_max = torch.maximum(self.qd_max, aq.amax(0))
    self.err_max = torch.maximum(self.err_max, ae.amax(0))
    self.qd_over3 += (aq > 3.0).sum(0)
    self.qd_over4 += (aq > 4.0).sum(0)
    self.tau_over_rated += (at > self.motor.tau_rated).sum(0)
    self.tau_near_cap += (at >= 0.95 * self.motor.tau_cap).sum(0)
    self.ep_qd_max = torch.maximum(self.ep_qd_max, aq.amax(-1))

  def end_episodes(self, env_ids) -> None:
    m = self.ep_qd_max[env_ids]
    self.eps += m.numel()
    self.eps_over4 += (m > 4.0).sum()
    self.ep_qd_max[env_ids] = 0.0

  def summary(self, joint_names: tuple[str, ...]) -> dict[str, float]:
    """Means, maxima and threshold fractions (one host transfer per call)."""
    n = float(self.n)
    if n == 0:
      return {}
    tau_mean = (self.tau_sum / n).tolist()
    qd_mean = (self.qd_sum / n).tolist()
    tau_max = self.tau_max.tolist()
    qd_max = self.qd_max.tolist()
    nj = self.nj
    out: dict[str, float] = {
      "tau_mean": sum(tau_mean) / nj,
      "tau_max": max(tau_max),
      "qd_mean": sum(qd_mean) / nj,
      "qd_max": max(qd_max),
      "err_mean": float(self.err_sum.sum()) / (n * nj),
      "err_max": float(self.err_max.max()),
      "qd_frac_above_3": float(self.qd_over3.sum()) / (n * nj),
      "qd_frac_above_4": float(self.qd_over4.sum()) / (n * nj),
      "tau_frac_above_rated": float(self.tau_over_rated.sum()) / (n * nj),
      "tau_frac_near_cap": float(self.tau_near_cap.sum()) / (n * nj),
      "slew_saturation_fraction": float(self.slew_sat) / max(float(self.slew_n), 1.0),
    }
    for j, name in enumerate(joint_names):
      short = name.replace("_joint", "")
      out[f"tau_mean/{short}"] = tau_mean[j]
      out[f"tau_max/{short}"] = tau_max[j]
      out[f"qd_mean/{short}"] = qd_mean[j]
      out[f"qd_max/{short}"] = qd_max[j]
    eps = float(self.eps)
    if eps > 0:
      out["episodes_with_qd_above_4"] = float(self.eps_over4) / eps
    return out


@dataclass(kw_only=True)
class FtsrActionCfg(ActionTermCfg):
  entity_name: str = "robot"
  joint_names: tuple[str, ...] = JOINT_NAMES
  scale: dict[str, float] = field(default_factory=dict)
  """Per joint-name regex; every joint must match exactly one entry."""
  raw_clip: float = 50.0
  slew_rate: float = 3.0
  """rad/s on the commanded target; 0 disables (not used by the reference tasks)."""
  passive_steps: int = 0
  motor: MotorEnvelopeCfg = field(default_factory=MotorEnvelopeCfg)
  assist: AssistCfg | None = None

  def build(self, env: ManagerBasedRlEnv) -> FtsrAction:
    return FtsrAction(self, env)


class FtsrAction(ActionTerm):
  cfg: FtsrActionCfg

  def __init__(self, cfg: FtsrActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ent = self._entity
    ids, names = ent.find_joints(cfg.joint_names, preserve_order=True)
    assert tuple(names) == tuple(cfg.joint_names), names
    self.joint_names = tuple(names)
    dev = self.device
    n, nj = self.num_envs, len(ids)
    self._ids = torch.tensor(ids, device=dev, dtype=torch.long)

    import re

    scale = torch.zeros(nj, device=dev)
    for j, name in enumerate(names):
      hits = [v for k, v in cfg.scale.items() if re.fullmatch(k, name)]
      assert len(hits) == 1, (name, hits)
      scale[j] = hits[0]
    self.scale = scale
    self.default = ent.data.default_joint_pos[0, self._ids].clone()
    self.q_min = torch.tensor([JOINT_RANGES[x][0] for x in names], device=dev)
    self.q_max = torch.tensor([JOINT_RANGES[x][1] for x in names], device=dev)
    self.slew_step = cfg.slew_rate * env.step_dt if cfg.slew_rate > 0 else float("inf")

    # Joint -> actuator (ctrl) index, for forcerange / gain writes.
    ctrl = torch.full((nj,), -1, dtype=torch.long)
    local = {int(i): j for j, i in enumerate(ids)}
    for act in ent.actuators:
      for t, c in zip(
        act.target_ids.tolist(), act.global_ctrl_ids.tolist(), strict=True
      ):
        if t in local:
          ctrl[local[t]] = c
    assert (ctrl >= 0).all()
    self._ctrl = ctrl.to(dev)

    self._raw = torch.zeros(n, nj, device=dev)
    self._prev_raw = torch.zeros_like(self._raw)
    self._prev_prev_raw = torch.zeros_like(self._raw)
    self.q_star = torch.zeros(n, nj, device=dev)
    self.q_cmd = torch.zeros(n, nj, device=dev)
    self.entered = torch.zeros(n, dtype=torch.bool, device=dev)
    self.passive = torch.zeros(n, dtype=torch.bool, device=dev)
    self._kp: torch.Tensor | None = None  # nominal gains, captured after startup DR
    self._kd: torch.Tensor | None = None

    body_ids, _ = ent.find_bodies(("base_link",))
    self._base = body_ids
    self.force = torch.zeros(n, 3, device=dev)
    self.torque = torch.zeros(n, 3, device=dev)
    self._cost = torch.zeros(n, 2, device=dev)
    env.extras[COST_KEY] = self._cost
    self._substeps = 0
    self.t_coeff = 0.0
    # Host-side assist state (no GPU reads in the physics loop): ``_assist_live``
    # while a nonzero wrench may be in the simulation; it is cleared with one zero
    # write in the first physics step at or after t_tag.
    self._assist_live = cfg.assist is not None
    # Per-policy-step effective gains (passive: kp 0, kd 1) and actuated mask.
    self._kp_eff = torch.zeros(n, nj, device=dev)
    self._kd_eff = torch.zeros(n, nj, device=dev)
    self._act_f = torch.ones(n, 1, device=dev)

    self.monitor = SubstepMonitor(nj, n, cfg.motor, dev)

  # ActionTerm interface.

  @property
  def action_dim(self) -> int:
    return len(self.joint_names)

  @property
  def raw_action(self) -> torch.Tensor:
    """a_c: raw-clipped network output (zero in the passive window)."""
    return self._raw

  @property
  def prev_raw_action(self) -> torch.Tensor:
    return self._prev_raw

  @property
  def prev_prev_raw_action(self) -> torch.Tensor:
    return self._prev_prev_raw

  def _nominal_gains(self) -> None:
    """Capture the per-env PD gains once (after startup randomization)."""
    if self._kp is None:
      model = self._env.sim.model
      self._kp = model.actuator_gainprm[:, self._ctrl, 0].clone()
      self._kd = -model.actuator_biasprm[:, self._ctrl, 2].clone()

  def _write_actuator_law(self, q: torch.Tensor, qd: torch.Tensor) -> None:
    """Select the active affine piece of tau = clip(tau_PD, L(qd), U(qd)).

    tau_PD = kp (q* - q) - kd qd (deploy gains, or kp 0 / kd 1 when passive).
    U(qd) = stall (1 - qd / w0), L(qd) = stall (-1 - qd / w0), both inside +-cap.
    Where the PD force exceeds a curve, the actuator law becomes that curve,
    tau = +-stall - (stall / w0) qd, written as an affine bias so that MuJoCo's
    implicitfast integrator treats its velocity slope implicitly; the forcerange is
    the static +-cap. The force equals ``motor_envelope`` clipping exactly; only the
    integration of the velocity dependence is implicit (an explicit per-step bound
    oscillates on the ankle-roll joint, whose inertia is 4.4e-4 kg m^2:
    dt stall / (w0 I) = 12 > 2).
    """
    m = self.cfg.motor
    kp, kd = self._kp_eff, self._kd_eff
    tau_pd = kp * (self.q_star - q) - kd * qd
    slope = m.tau_stall / m.omega0
    upper = m.tau_stall - slope * qd
    lower = -m.tau_stall - slope * qd
    on_upper = tau_pd > upper
    on_lower = tau_pd < lower
    curve = on_upper | on_lower
    model = self._env.sim.model
    c = self._ctrl
    model.actuator_gainprm[:, c, 0] = torch.where(curve, 0.0, kp)
    model.actuator_biasprm[:, c, 0] = (
      on_upper.float() - on_lower.float()
    ) * m.tau_stall
    model.actuator_biasprm[:, c, 1] = torch.where(curve, 0.0, -kp)
    model.actuator_biasprm[:, c, 2] = torch.where(curve, -slope, -kd)

  def process_actions(self, actions: torch.Tensor) -> None:
    env = self._env
    q = self._entity.data.joint_pos[:, self._ids]
    k = env.episode_length_buf
    passive = (k < self.cfg.passive_steps) & ~self.entered
    entry = ~passive & ~self.entered
    q_meas = torch.clamp(q, self.q_min, self.q_max)
    self.q_star = torch.where(entry.unsqueeze(-1), q_meas, self.q_star)
    self.entered |= entry
    self.passive = passive
    self._nominal_gains()
    assert self._kp is not None and self._kd is not None
    p = passive.unsqueeze(-1)
    self._kp_eff = torch.where(p, 0.0, self._kp)
    self._kd_eff = torch.where(p, PASSIVE_KD, self._kd)
    self._act_f = (~p).float()

    a = torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)
    a = torch.where(passive.unsqueeze(-1), torch.zeros_like(a), a)
    self._prev_prev_raw = self._prev_raw
    self._prev_raw = self._raw
    self._raw = a

    q_cmd = torch.clamp(self.default + self.scale * a, self.q_min, self.q_max)
    delta = q_cmd - self.q_star
    step = self.slew_step
    q_new = self.q_star + torch.clamp(delta, -step, step)
    act = ~passive
    self.q_star = torch.where(passive.unsqueeze(-1), q_meas, q_new)
    self.q_cmd = q_cmd

    sat = (delta.abs() > step + 1e-6) & act.unsqueeze(-1)
    self.monitor.slew_sat += sat.sum()
    self.monitor.slew_n += act.sum() * delta.shape[1]

    self._cost.zero_()
    self._substeps = 0
    a_cfg = self.cfg.assist
    if a_cfg is not None:
      self.t_coeff = time_coeff(env.common_step_counter, a_cfg)
      if env.common_step_counter >= a_cfg.t_tag:
        assert self.t_coeff == 0.0

  def apply_actions(self) -> None:
    env = self._env
    data = self._entity.data
    qd = data.joint_vel[:, self._ids]
    q = data.joint_pos[:, self._ids]

    if self._substeps > 0:
      # Results of the previous physics step of this policy step.
      tau = data.qfrc_actuator[:, self._ids]
      self.monitor.add(tau, qd, self.q_star - q, self._act_f)

    self._write_actuator_law(q, qd)
    self._entity.set_joint_position_target(self.q_star, joint_ids=self._ids)

    a_cfg = self.cfg.assist
    if a_cfg is not None and self._assist_live:
      if self.t_coeff > 0.0:
        stage = env.extras[STAGE_KEY]
        h = data.body_link_pos_w[:, self._base[0], 2]
        quat = data.body_link_quat_w[:, self._base[0]]
        f, t = eq4_wrench(h, quat, stage.h_cmd, self.t_coeff, a_cfg)
        self.force, self.torque = f * self._act_f, t * self._act_f
        # Normalized constraint costs (Eq. 5-8 units: F / F_max, T / (T_max pi)),
        # averaged over the policy step's physics steps.
        d = float(env.cfg.decimation)
        self._cost[:, 0] += self.force.norm(dim=-1) / a_cfg.f_max / d
        self._cost[:, 1] += self.torque.norm(dim=-1) / (a_cfg.t_max * torch.pi) / d
      else:
        # t >= t_tag: clear the wrench once; costs stay exactly zero.
        self.force.zero_()
        self.torque.zero_()
        self._assist_live = False
      self._entity.write_external_wrench_to_sim(
        self.force[:, None, :], self.torque[:, None, :], body_ids=self._base
      )
    self._substeps += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    self._prev_raw[env_ids] = 0.0
    self._prev_prev_raw[env_ids] = 0.0
    self.entered[env_ids] = False
    self.monitor.end_episodes(env_ids)
    if self.cfg.assist is not None and self._assist_live:
      self.force[env_ids] = 0.0
      self.torque[env_ids] = 0.0
      self._entity.write_external_wrench_to_sim(
        self.force[:, None, :], self.torque[:, None, :], body_ids=self._base
      )

  def log_assist(self) -> dict[str, float | torch.Tensor]:
    """Device tensors (and the host time coefficient); reduced by the caller once
    per iteration."""
    f = self.force.norm(dim=-1)
    t = self.torque.norm(dim=-1)
    return {
      "Assist/time_coeff": self.t_coeff,
      "Assist/force_mean": f.mean(),
      "Assist/force_max": f.max(),
      "Assist/torque_mean": t.mean(),
      "Assist/torque_max": t.max(),
    }
