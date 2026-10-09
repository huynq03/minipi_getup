"""Joint-position action with the Mini-Pi plant (experiment pd16_noslew).

Per policy step (``process_actions``)::

    a_c    = clip(a, -raw_clip, raw_clip)                 raw clip
    q_cmd  = q_default + scale * a_c                      absolute target
    q*     = clip(q_cmd, q_min, q_max)                    physical per-joint clip
    PD     tau = clip(kp (q* - q) - kd qdot, -16, 16)     native position actuator,
                                                          held 40 x 0.5 ms

- No target slew / rate limit (removed in this experiment), no torque-speed
  derating. Measured qdot and the tracking error q* - q are never clamped.
- ``last_action`` (observation) is the raw-clipped network output ``a_c``, zero at
  reset and during the passive window (deploy semantics).
- Passive window (``passive_steps`` after a reset, recovery task only): the deploy
  Passive state, kp 0 / kd 1 on every joint; actions, observations and assistance
  off. The effective gains are written to the actuator once per policy step.

Per physics step (``apply_actions``): Eq. 4 assistance (``assistance.py``) if
configured, and the substep monitor (torque, qdot, tracking error).

Non-finite physics (a simulation explosion in one env): the env is flagged in
``env.extras[INVALID_KEY]`` for the policy step, its wrench and constraint costs are
set to zero at the source and it is left out of the monitor; mjlab's ``nan``
termination resets it. Finite envs are bitwise unchanged (``torch.where``).
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
  TAU_CAP,
  TAU_RATED_REPORT,
)
from minipi_getup.ftsr_ref.mdp.assistance import AssistCfg, eq4_wrench, time_coeff
from minipi_getup.ftsr_ref.mdp.stages import STAGE_KEY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

COST_KEY = "ftsr_constraint_cost"
# Per-env bool: the physics state, a joint reading or the Eq. 4 wrench was non-finite
# in some physics step of the current policy step.
INVALID_KEY = "ftsr_invalid_env"
# Per-env max finite |qdot| over the current policy step (diagnostics of explosions).
QD_STEP_MAX_KEY = "ftsr_qd_step_max"


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def plant_setup(env: ManagerBasedRlEnv, env_ids) -> None:
  """Startup event: makes the actuator gain fields per-env (passive gains are
  written per env by the action term)."""
  del env, env_ids


class SubstepMonitor:
  """Physics-step actuator / speed statistics of actuated envs, for training logs.

  Cheap sufficient statistics only (per joint): sample count, sum and max of |tau|,
  |qdot| and |q* - q|, and threshold counts (|qdot| > 3, > 4 and > ``qd_soft`` rad/s,
  |tau| above the 6 Nm reporting threshold, |tau| >= 0.95 cap). Everything stays on the device during
  the rollout; ``summary()`` converts once per iteration. Quantiles (p95 / p99) are
  not computed here: ``evaluate.py`` reports them from full histograms.
  """

  def __init__(self, nj: int, num_envs: int, device, qd_soft: float = 6.28):
    self.qd_soft = qd_soft
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
    self.qd_over3, self.qd_over4, self.qd_over_soft = z(nj), z(nj), z(nj)
    self.tau_over_rated, self.tau_near_cap = z(nj), z(nj)
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
    self.qd_over_soft += (aq > self.qd_soft).sum(0)
    self.tau_over_rated += (at > TAU_RATED_REPORT).sum(0)
    self.tau_near_cap += (at >= 0.95 * TAU_CAP).sum(0)
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
      "qd_frac_above_soft": float(self.qd_over_soft.sum()) / (n * nj),
      "tau_frac_near_cap": float(self.tau_near_cap.sum()) / (n * nj),
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
  passive_steps: int = 0
  qd_soft: float = 6.28
  """Monitoring threshold for |qdot| (= the qd_soft_envelope reward limit)."""
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

    # Joint -> actuator (ctrl) index, for the passive-gain writes.
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
    self._invalid = torch.zeros(n, dtype=torch.bool, device=dev)
    self._qd_step_max = torch.zeros(n, device=dev)
    env.extras[INVALID_KEY] = self._invalid
    env.extras[QD_STEP_MAX_KEY] = self._qd_step_max
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

    self.monitor = SubstepMonitor(nj, n, dev, cfg.qd_soft)

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

  def _write_gains(self) -> None:
    """Write the effective PD law (deploy gains, or kp 0 / kd 1 when passive) as the
    position actuator's affine law: tau = kp (q* - q) - kd qdot, clipped by the
    static forcerange +-TAU_CAP. Once per policy step (gains only change there)."""
    model = self._env.sim.model
    c = self._ctrl
    model.actuator_gainprm[:, c, 0] = self._kp_eff
    model.actuator_biasprm[:, c, 0] = 0.0
    model.actuator_biasprm[:, c, 1] = -self._kp_eff
    model.actuator_biasprm[:, c, 2] = -self._kd_eff

  def process_actions(self, actions: torch.Tensor) -> None:
    env = self._env
    q = self._entity.data.joint_pos[:, self._ids]
    k = env.episode_length_buf
    passive = (k < self.cfg.passive_steps) & ~self.entered
    entry = ~passive & ~self.entered
    q_meas = torch.clamp(q, self.q_min, self.q_max)
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
    # No slew: the clipped command is the PD target (passive: hold the measured pose,
    # irrelevant with kp 0).
    self.q_star = torch.where(passive.unsqueeze(-1), q_meas, q_cmd)
    self.q_cmd = q_cmd
    self._write_gains()

    self._cost.zero_()
    self._invalid.zero_()
    self._qd_step_max.zero_()
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

    # Envs whose physics state went non-finite are flagged and kept out of the
    # monitor and the assistance (identity for finite rows).
    ok = torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1)
    torch.maximum(
      self._qd_step_max,
      torch.nan_to_num(qd.abs(), nan=0.0, posinf=0.0).amax(-1),
      out=self._qd_step_max,
    )
    if self._substeps > 0:
      # Results of the previous physics step of this policy step.
      tau = data.qfrc_actuator[:, self._ids]
      ok_tau = ok & torch.isfinite(tau).all(-1)
      self._invalid |= ~ok_tau
      okc = ok_tau.unsqueeze(-1)
      self.monitor.add(
        torch.where(okc, tau, 0.0),
        torch.where(okc, qd, 0.0),
        torch.where(okc, self.q_star - q, 0.0),
        self._act_f * okc,
      )
    else:
      self._invalid |= ~ok

    self._entity.set_joint_position_target(self.q_star, joint_ids=self._ids)

    a_cfg = self.cfg.assist
    if a_cfg is not None and self._assist_live:
      if self.t_coeff > 0.0:
        stage = env.extras[STAGE_KEY]
        h = data.body_link_pos_w[:, self._base[0], 2]
        quat = data.body_link_quat_w[:, self._base[0]]
        f, t = eq4_wrench(h, quat, stage.h_cmd, self.t_coeff, a_cfg)
        f, t = f * self._act_f, t * self._act_f
        # Source sanitation: a non-finite base state gives a non-finite wrench; such
        # an env gets no wrench and zero cost (and is flagged invalid).
        w_ok = torch.isfinite(f).all(-1) & torch.isfinite(t).all(-1)
        self._invalid |= ~w_ok
        self.force = torch.where(w_ok.unsqueeze(-1), f, 0.0)
        self.torque = torch.where(w_ok.unsqueeze(-1), t, 0.0)
        # Normalized constraint costs (Eq. 5-8 units: F / F_max, T / (T_max pi)),
        # averaged over the policy step's physics steps.
        d = float(env.cfg.decimation)
        self._cost[:, 0] += self.force.norm(dim=-1) / a_cfg.f_max / d
        self._cost[:, 1] += self.torque.norm(dim=-1) / (a_cfg.t_max * torch.pi) / d
        # An invalid env's sample carries no cost (also the substeps before the
        # explosion); finite envs are untouched.
        self._cost.masked_fill_(self._invalid.unsqueeze(-1), 0.0)
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
