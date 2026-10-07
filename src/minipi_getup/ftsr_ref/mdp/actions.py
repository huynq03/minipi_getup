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


class _Hist:
  """Per-joint histogram of |x| with fixed bins (quantiles without storing samples)."""

  def __init__(self, nj: int, width: float, vmax: float, device):
    self.nj, self.width = nj, width
    self.nb = int(round(vmax / width))
    self.device = device
    self.clear()

  def clear(self) -> None:
    self.counts = torch.zeros(self.nj, self.nb, device=self.device, dtype=torch.float64)
    self.total = torch.zeros(self.nj, device=self.device, dtype=torch.float64)
    self.max = torch.zeros(self.nj, device=self.device)
    self.n = torch.zeros(self.nj, device=self.device, dtype=torch.float64)

  def add(self, x: torch.Tensor) -> None:
    """x: (B, nj), already masked to valid samples."""
    if x.numel() == 0:
      return
    a = x.abs()
    idx = (a / self.width).long().clamp_(max=self.nb - 1)
    flat = idx + torch.arange(self.nj, device=self.device) * self.nb
    self.counts += (
      torch.bincount(flat.flatten(), minlength=self.nj * self.nb)
      .double()
      .view(self.nj, self.nb)
    )
    self.total += a.sum(0).double()
    self.max = torch.maximum(self.max, a.amax(0))
    self.n += a.shape[0]

  def frac_above(self, thr: float) -> torch.Tensor:
    i = int(thr / self.width)
    return self.counts[:, i:].sum(1) / self.n.clamp(min=1)

  def quantile(self, q: float) -> torch.Tensor:
    cdf = torch.cumsum(self.counts, 1) / self.n.clamp(min=1).unsqueeze(1)
    idx = (cdf < q).sum(1)
    return (idx + 1).float() * self.width

  def summary(self, prefix: str) -> dict[str, torch.Tensor]:
    return {
      f"{prefix}_mean": (self.total / self.n.clamp(min=1)).float(),
      f"{prefix}_p95": self.quantile(0.95),
      f"{prefix}_p99": self.quantile(0.99),
      f"{prefix}_max": self.max,
    }


class SubstepMonitor:
  """Physics-substep actuator / speed statistics of actuated envs."""

  def __init__(self, nj: int, num_envs: int, motor: MotorEnvelopeCfg, device):
    self.motor = motor
    self.device = device
    self.tau = _Hist(nj, 0.02, 40.0, device)
    self.qd = _Hist(nj, 0.02, 60.0, device)
    self.err = _Hist(nj, 0.002, 4.0, device)
    self.ep_qd_max = torch.zeros(num_envs, device=device)
    self.clear()

  def clear(self) -> None:
    self.tau.clear()
    self.qd.clear()
    self.err.clear()
    self.slew_sat = 0.0
    self.slew_n = 0.0
    self.eps = 0.0
    self.eps_over4 = 0.0
    self.near_cap = torch.zeros((), device=self.device, dtype=torch.float64)

  def add(self, tau, qd, err, mask) -> None:
    self.tau.add(tau[mask])
    self.qd.add(qd[mask])
    self.err.add(err[mask])
    self.near_cap += (tau[mask].abs() >= 0.95 * self.motor.tau_cap).sum().double()
    self.ep_qd_max = torch.where(
      mask, torch.maximum(self.ep_qd_max, qd.abs().amax(-1)), self.ep_qd_max
    )

  def end_episodes(self, env_ids) -> None:
    m = self.ep_qd_max[env_ids]
    self.eps += float(m.numel())
    self.eps_over4 += float((m > 4.0).sum())
    self.ep_qd_max[env_ids] = 0.0

  def summary(self, joint_names: tuple[str, ...]) -> dict[str, float]:
    """Scalars: whole-robot aggregates plus per-joint p99/max."""
    out: dict[str, float] = {}
    n = float(self.tau.n.sum())
    if n == 0:
      return out
    for prefix, h in (("tau", self.tau), ("qd", self.qd), ("err", self.err)):
      s = h.summary(prefix)
      for k, v in s.items():
        out[k] = (
          float(v.max()) if k.endswith(("max", "p95", "p99")) else float(v.mean())
        )
      for j, name in enumerate(joint_names):
        short = name.replace("_joint", "")
        out[f"{prefix}_p99/{short}"] = float(s[f"{prefix}_p99"][j])
        out[f"{prefix}_max/{short}"] = float(s[f"{prefix}_max"][j])
    qd_counts = self.qd.counts.sum(0)
    w = self.qd.width
    out["qd_frac_above_3"] = float(qd_counts[int(3.0 / w) :].sum() / n)
    out["qd_frac_above_4"] = float(qd_counts[int(4.0 / w) :].sum() / n)
    tc = self.tau.counts.sum(0)
    out["tau_frac_above_rated"] = float(
      tc[int(self.motor.tau_rated / self.tau.width) :].sum() / n
    )
    out["tau_frac_near_cap"] = float(self.near_cap / n)
    out["slew_saturation_fraction"] = self.slew_sat / max(self.slew_n, 1.0)
    if self.eps > 0:
      out["episodes_with_qd_above_4"] = self.eps_over4 / self.eps
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
    assert self._kp is not None and self._kd is not None
    m = self.cfg.motor
    p = self.passive.unsqueeze(-1)
    kp = torch.where(p, torch.zeros_like(self._kp), self._kp)
    kd = torch.where(p, torch.full_like(self._kd, PASSIVE_KD), self._kd)
    tau_pd = kp * (self.q_star - q) - kd * qd
    slope = m.tau_stall / m.omega0
    upper = m.tau_stall - slope * qd
    lower = -m.tau_stall - slope * qd
    on_upper = tau_pd > upper
    on_lower = tau_pd < lower
    curve = on_upper | on_lower
    model = self._env.sim.model
    c = self._ctrl
    model.actuator_gainprm[:, c, 0] = torch.where(curve, torch.zeros_like(kp), kp)
    model.actuator_biasprm[:, c, 0] = torch.where(
      on_upper,
      torch.full_like(kp, m.tau_stall),
      torch.where(on_lower, torch.full_like(kp, -m.tau_stall), torch.zeros_like(kp)),
    )
    model.actuator_biasprm[:, c, 1] = torch.where(curve, torch.zeros_like(kp), -kp)
    model.actuator_biasprm[:, c, 2] = torch.where(
      curve, torch.full_like(kd, -slope), -kd
    )

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

    n_act = float(act.sum())
    if n_act > 0:
      sat = (delta.abs() > step + 1e-6) & act.unsqueeze(-1)
      self.monitor.slew_sat += float(sat.sum())
      self.monitor.slew_n += n_act * delta.shape[1]

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
    act = ~self.passive

    if self._substeps > 0:
      # Results of the previous physics step of this policy step.
      tau = data.qfrc_actuator[:, self._ids]
      err = self.q_star - data.joint_pos[:, self._ids]
      self.monitor.add(tau, qd, err, act)

    self._write_actuator_law(data.joint_pos[:, self._ids], qd)
    self._entity.set_joint_position_target(self.q_star, joint_ids=self._ids)

    a_cfg = self.cfg.assist
    if a_cfg is not None and (self.t_coeff > 0.0 or bool(self.force.any())):
      if self.t_coeff > 0.0:
        stage = env.extras[STAGE_KEY]
        h = data.body_link_pos_w[:, self._base[0], 2]
        quat = data.body_link_quat_w[:, self._base[0]]
        f, t = eq4_wrench(h, quat, stage.h_cmd, self.t_coeff, a_cfg)
        m = act.float().unsqueeze(-1)
        self.force, self.torque = f * m, t * m
      else:
        self.force.zero_()
        self.torque.zero_()
      self._entity.write_external_wrench_to_sim(
        self.force[:, None, :], self.torque[:, None, :], body_ids=self._base
      )
      # Normalized constraint costs (Eq. 5-8 units: F / F_max, T / (T_max pi)),
      # averaged over the policy step's physics steps.
      d = float(env.cfg.decimation)
      self._cost[:, 0] += self.force.norm(dim=-1) / a_cfg.f_max / d
      self._cost[:, 1] += self.torque.norm(dim=-1) / (a_cfg.t_max * torch.pi) / d
    self._substeps += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    self._prev_raw[env_ids] = 0.0
    self._prev_prev_raw[env_ids] = 0.0
    self.entered[env_ids] = False
    self.monitor.end_episodes(env_ids)
    if self.cfg.assist is not None:
      self.force[env_ids] = 0.0
      self.torque[env_ids] = 0.0
      self._entity.write_external_wrench_to_sim(
        self.force[:, None, :], self.torque[:, None, :], body_ids=self._base
      )

  def log_assist(self) -> dict[str, float]:
    f = self.force.norm(dim=-1)
    t = self.torque.norm(dim=-1)
    return {
      "Assist/time_coeff": self.t_coeff,
      "Assist/force_mean": float(f.mean()),
      "Assist/force_max": float(f.max()),
      "Assist/torque_mean": float(t.mean()),
      "Assist/torque_max": float(t.max()),
    }
