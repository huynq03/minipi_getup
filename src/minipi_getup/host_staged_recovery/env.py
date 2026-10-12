"""Staged recovery environment: task-space pipeline shaping on host_safe_recovery."""

from __future__ import annotations

import math

import torch
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg

from minipi_getup.host.env import torch_rand_float
from minipi_getup.host_safe_recovery.env import (
  SafeRecoveryEnv,
  _saturating_support,
  _sole_tilt_cost,
)
from minipi_getup.host_staged_recovery import geometry as geo

LEG_CONTACT_SENSOR = "staged_leg_leg_contact"


def weight_of(env) -> torch.Tensor:
  body_ids = env.robot.indexing.body_ids.long()
  return env.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(env.cfg.sim.gravity[2])


class StagedRecoveryEnv(SafeRecoveryEnv):
  """Legs in -> feet down -> COM over support -> rise -> default posture.

  Shared geometry (whole-body COM, sole corners, support polygon) is computed
  once per policy step in `_update_stage_state`, before any reward term reads it.
  """

  def _extra_sensors(self) -> tuple:
    # Net force between the whole left and right leg subtrees (shells included).
    return (ContactSensorCfg(
      name=LEG_CONTACT_SENSOR,
      primary=ContactMatch(mode="subtree", pattern="l_hip_pitch_link", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="r_hip_pitch_link", entity="robot"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=self.cfg.control.decimation,
    ),)

  def _init_buffers(self):
    super()._init_buffers()
    self.leg_contact_sensor: ContactSensor = self.scene[LEG_CONTACT_SENSOR]
    n, dev = self.num_envs, self.device
    names = list(self.robot.body_names)
    self.hip_body_indices = torch.tensor(
      [names.index(f"{s}_hip_pitch_link") for s in ("l", "r")], device=dev
    )
    span = self.cfg.staged.reset_half_span
    self.reset_half_span = torch.tensor(
      [span[next(k for k in span if j.endswith(f"_{k}_joint"))] for j in self.dof_names],
      device=dev,
    )
    zeros = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
    self.com_w = zeros(3)
    self.com_prev = zeros(3)
    self.com_prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)
    self.com_vz = zeros()
    self.com_outside = zeros()
    self.com_score = zeros()
    self.sole_centres = zeros(2, 3)
    self.stance_width = zeros()
    self.foot_gap = zeros()
    self.leg_contact_force = zeros()
    self.leg_contact = zeros()
    self.ep_leg_contact_time = zeros()
    self.ep_min_foot_share = zeros()
    self.prev_height = zeros()
    self.foot_flat = zeros(2)
    self.knee_support = zeros()
    self.knee_time = zeros()
    self.lift_gate = zeros()
    # Episode monitors.
    self.t_feet = torch.full((n,), -1.0, device=dev)
    self.t_com = torch.full((n,), -1.0, device=dev)
    self.t_rise = torch.full((n,), -1.0, device=dev)
    self.rise_supported = torch.zeros(n, dtype=torch.bool, device=dev)
    self.ep_knee_time = zeros()
    self.ep_reverse_standing = zeros()
    self.ep_standing_steps = zeros()
    self.ep_com_err = zeros()
    self.ep_com_err_steps = zeros()
    self.actuated_peak_vel = zeros()
    self.step_peak_qd = zeros(self.num_dof)

  # ------------------------------------------------------------------ geometry
  def _update_stage_state(self):
    cfg = self.cfg.staged
    d = self.robot.data
    self.com_w[:] = d.data.subtree_com[:, self.robot.indexing.root_body_id]
    vz = (self.com_w[:, 2] - self.com_prev[:, 2]) / self.dt
    self.com_vz[:] = torch.where(self.com_prev_valid, vz, torch.zeros_like(vz))
    self.com_prev.copy_(self.com_w)
    self.com_prev_valid.fill_(True)

    foot_pos = d.body_link_pos_w[:, self.sole_body_indices]
    foot_quat = d.body_link_quat_w[:, self.sole_body_indices]
    corners = geo.sole_corners_world(foot_pos, foot_quat)
    self.sole_centres[:] = corners.mean(2)
    left, right, _ = self._normalized_supports()
    load = torch.stack((left, right), dim=1)
    valid = (corners[..., 2] < cfg.corner_contact_height_m) & (load[:, :, None] > cfg.support_foot_min)
    outside, centre, any_valid = geo.support_polygon_distance(
      self.com_w[:, :2], corners[..., :2].reshape(-1, 8, 2), valid.reshape(-1, 8)
    )
    self.com_outside[:] = torch.where(any_valid, outside, torch.full_like(outside, math.nan))
    self.com_score[:] = geo.com_support_score(
      outside, centre, any_valid, cfg.com_support_sigma_m, cfg.com_centre_sigma_m
    )
    delta = self.sole_centres[:, 0, :2] - self.sole_centres[:, 1, :2]
    self.stance_width[:] = torch.linalg.vector_norm(delta, dim=-1)
    self.foot_gap[:] = geo.foot_lateral_gap(corners, geo.lateral_axis(self.root_states[:, 3:7], foot_quat))
    self.foot_flat[:] = 1.0 - _sole_tilt_cost(foot_quat, self.cfg.safe_recovery.sole_tilt_sigma_rad)
    self.lift_gate[:] = torch.sigmoid(
      (self.root_states[:, 2] - cfg.lift_gate_height_center) / cfg.lift_gate_height_width
    )

    legs = self.leg_contact_sensor.data.force_history  # (N, 1, H, 3)
    self.leg_contact_force[:] = torch.linalg.vector_norm(legs, dim=-1).mean(dim=-1).sum(dim=1)
    self.leg_contact[:] = _saturating_support(
      self.leg_contact_force, weight_of(self), cfg.leg_contact_deadband_weight_fraction,
      cfg.leg_contact_scale_weight_fraction,
    )
    history = self.nonfoot_ground_sensor.data.force_history
    knee = history[:, self.knee_force_indices, :, 2].abs().mean(dim=2).sum(dim=1)
    body_ids = self.robot.indexing.body_ids.long()
    weight = self.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(self.cfg.sim.gravity[2])
    sr = self.cfg.safe_recovery
    self.knee_support[:] = _saturating_support(
      knee, weight, sr.contact_force_deadband_weight_fraction, sr.contact_force_scale_weight_fraction
    )
    on_knee = self.knee_support > cfg.knee_contact_support_min
    decay = math.exp(-self.dt / cfg.knee_contact_decay_s)
    self.knee_time[:] = torch.where(on_knee, self.knee_time + self.dt, self.knee_time * decay)

    self._update_stage_monitors(load)

  def _update_stage_monitors(self, load):
    cfg = self.cfg.staged
    actuated = self.real_episode_length_buf > self.unactuated_time
    t = (self.real_episode_length_buf.float() - self.unactuated_time) * self.dt
    sustained = (load * self.support_memory).min(dim=1).values
    feet = actuated & (sustained > cfg.stage_feet_support)
    stance = geo.adequate_stance(self.foot_gap, self.stance_width, cfg.stance_min_gap_m, cfg.stance_max_width_m)
    supported = feet & stance & (self.com_outside <= cfg.stage_com_supported_m)
    com = supported & (self.com_score > cfg.stage_com_score) & (self.lift_gate > 0.5)
    for buf, hit in ((self.t_feet, feet), (self.t_com, com)):
      buf[:] = torch.where(hit & (buf < 0), t, buf)
    # Rise onset: last upward crossing of the rise height before first standing,
    # with the support state at that instant.
    h = self.root_states[:, 2]
    crossing = actuated & ~self.ever_stood & (h > cfg.stage_rise_height_m) & (
      self.prev_height <= cfg.stage_rise_height_m
    ) & (-self.projected_gravity[:, 2] > cfg.stage_rise_upright)
    self.t_rise[:] = torch.where(crossing, t, self.t_rise)
    self.rise_supported[:] = torch.where(crossing, supported, self.rise_supported)
    self.prev_height.copy_(h)
    standing = self._standing_gate() > 0.5
    self.ep_knee_time += (self.knee_support > cfg.knee_contact_support_min).float() * self.dt * actuated
    self.ep_leg_contact_time += (self.leg_contact > 0.1).float() * self.dt * actuated
    self.ep_standing_steps += standing.float()
    force_l, force_r, _ = self._ground_support_forces()
    share = torch.minimum(force_l, force_r) / (force_l + force_r).clamp_min(1.0)
    self.ep_min_foot_share += share * standing
    calf = self.dof_pos[:, self.knee_contact_indices]
    self.ep_reverse_standing += (standing & (calf < -cfg.knee_reverse_soft_rad).any(1)).float()
    err_mask = actuated & (self.lift_gate > 0.5) & torch.isfinite(self.com_outside)
    self.ep_com_err += torch.where(err_mask, self.com_outside, torch.zeros_like(self.com_outside))
    self.ep_com_err_steps += err_mask.float()

  def compute_reward(self):
    self._update_stage_state()
    super().compute_reward()
    self.step_peak_qd.zero_()  # accumulates over the next step's substeps

  # ------------------------------------------------------------------ rewards
  def _reward_feet_under_com(self):
    cfg = self.cfg.staged
    return geo.feet_under_com_score(
      self.sole_centres[..., :2], self.com_w[:, :2],
      cfg.feet_under_com_radius_m, cfg.feet_under_com_scale_m,
    )

  def _reward_two_foot_flat_support(self):
    left, right, _ = self._normalized_supports()
    return left * self.foot_flat[:, 0] * right * self.foot_flat[:, 1]

  def _reward_com_support(self):
    left, right, _ = self._normalized_supports()
    return self.com_score * torch.maximum(left, right) * self.lift_gate

  def _reward_stance_width(self):
    cfg = self.cfg.staged
    left, right, _ = self._normalized_supports()
    cost = geo.stance_width_cost(
      self.foot_gap, self.stance_width, cfg.stance_min_gap_m, cfg.stance_max_width_m,
      cfg.stance_gap_scale_m, cfg.stance_width_scale_m, cfg.stance_cost_cap,
    )
    # Gate on either foot: one foot resting on or crossing the other also pays.
    return cost * torch.maximum(left, right) * self.lift_gate

  def _reward_unsupported_ascent(self):
    cfg = self.cfg.staged
    actuated = (self.real_episode_length_buf > self.unactuated_time).float()
    return geo.unsupported_ascent_cost(
      self.com_vz, self.com_score, self.lift_gate * actuated,
      cfg.unsupported_ascent_threshold_m_s, cfg.unsupported_ascent_scale_m_s,
    )

  def _reward_knee_reverse_lock(self):
    cfg = self.cfg.staged
    left, right, _ = self._normalized_supports()
    # Loaded leg, or any leg while standing (an unloaded standing leg may not lock).
    weight = torch.maximum(torch.stack((left, right), dim=1), self._standing_gate()[:, None])
    return geo.knee_reverse_lock_cost(
      self.dof_pos[:, self.knee_contact_indices], weight,
      cfg.knee_reverse_soft_rad, cfg.knee_reverse_span_rad,
    )

  def _reward_leg_contact(self):
    # Legs pressing on each other (feet stacked, knees/shins/thighs touching).
    actuated = (self.real_episode_length_buf > self.unactuated_time).float()
    return self.leg_contact * self.lift_gate * actuated

  def _reward_standing_load_imbalance(self):
    left, right, _ = self._ground_support_forces()
    body_ids = self.robot.indexing.body_ids.long()
    weight = self.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(self.cfg.sim.gravity[2])
    cost = geo.load_imbalance_cost(left, right, weight, self.cfg.staged.load_imbalance_floor_weight_fraction)
    return cost * self._standing_gate()

  def _reward_knee_load_support(self):
    cfg = self.cfg.staged
    ramp = geo.knee_contact_ramp(
      self.knee_time, cfg.knee_contact_ramp_s, cfg.knee_contact_initial_fraction
    )
    return super()._reward_knee_load_support() * ramp

  def _com_height_credit(self):
    floor = self.cfg.staged.height_credit_com_floor
    return floor + (1.0 - floor) * self.com_score

  def _knee_height_credit(self):
    # Used by head height (task) and target base height: rising credit needs
    # feet-supported COM, keeping a floor so early exploration still gets signal.
    return super()._knee_height_credit() * self._com_height_credit()

  def _reward_foot_supported_height(self):
    return super()._reward_foot_supported_height() * self._com_height_credit()

  # ------------------------------------------------------------------ actuation
  def _compute_torques(self, actions):
    torques = super()._compute_torques(actions)
    cfg = self.cfg.staged
    return geo.speed_limited_torque(
      torques, self.dof_vel, self.torque_limits.unsqueeze(0),
      cfg.motor_max_speed_rad_s, cfg.motor_speed_band_rad_s,
    )

  def _reward_joint_overspeed(self):
    # Same cost as host_safe_recovery, but on the 5 ms substep peak of this
    # policy step instead of the end-of-step velocity, so short spikes count.
    excess = torch.relu(self.step_peak_qd - self.cfg.safe_recovery.overspeed_soft_rad_s).square()
    return (0.5 * excess.mean(dim=1) + 0.5 * excess.max(dim=1).values) * self._recovery_regu_multiplier()

  def _substep_monitor(self):
    # Contact sensors here are stepped manually, like host_safe_recovery's own.
    self.leg_contact_sensor.update(self.cfg.sim.dt)
    super()._substep_monitor()
    self.step_peak_qd = torch.maximum(self.step_peak_qd, self.dof_vel.abs())
    actuated = self.real_episode_length_buf > self.unactuated_time
    peak = self.dof_vel.abs().max(dim=1).values * actuated
    self.actuated_peak_vel = torch.maximum(self.actuated_peak_vel, peak)

  # ------------------------------------------------------------------ resets
  def _reset_dofs(self, env_ids):
    """Baseline reset distribution with absolute half-spans (ROM-independent)."""
    n = len(env_ids)
    lower = self.robot.data.joint_pos_limits[0, :, 0]
    upper = self.robot.data.joint_pos_limits[0, :, 1]
    center = self.default_dof_pos.squeeze(0)
    q = center + torch_rand_float(-1.0, 1.0, (n, self.num_dof), self.device) * self.reset_half_span
    margin = 0.02
    self.dof_pos[env_ids] = torch.maximum(
      torch.minimum(q, upper.unsqueeze(0) - margin), lower.unsqueeze(0) + margin
    )
    self.dof_vel[env_ids] = 0.0
    self.robot.write_joint_state_to_sim(
      self.dof_pos[env_ids], self.dof_vel[env_ids], env_ids=env_ids
    )

  def _log_reset_metrics(self, env_ids):
    # The HoST monitor clears ever_stood inside its own logging.
    stood = self.ever_stood[env_ids].clone()
    super()._log_reset_metrics(env_ids)
    if not len(env_ids):
      return
    s = self.reset_stats
    t_feet, t_com, t_rise = self.t_feet[env_ids], self.t_com[env_ids], self.t_rise[env_ids]

    def frac(mask):
      return mask[stood].float().mean() if stood.any() else torch.tensor(math.nan)

    def mean_t(t):
      ok = stood & (t >= 0)
      return t[ok].mean() if ok.any() else torch.tensor(math.nan)

    s["stage_feet_before_rise"] = frac((t_feet >= 0) & ((t_rise < 0) | (t_feet <= t_rise)))
    s["stage_com_before_rise"] = frac((t_com >= 0) & ((t_rise < 0) | (t_com <= t_rise)))
    s["stage_supported_at_rise"] = frac(self.rise_supported[env_ids])
    s["stage_t_feet"] = mean_t(t_feet)
    s["stage_t_com"] = mean_t(t_com)
    s["stage_t_rise"] = mean_t(t_rise)
    s["knee_contact_time_s"] = self.ep_knee_time[env_ids].mean()
    s["leg_contact_time_s"] = self.ep_leg_contact_time[env_ids].mean()
    s["reverse_knee_standing_frac"] = (
      self.ep_reverse_standing[env_ids] / self.ep_standing_steps[env_ids].clamp_min(1.0)
    ).mean()
    s["standing_min_foot_share"] = (
      self.ep_min_foot_share[env_ids] / self.ep_standing_steps[env_ids].clamp_min(1.0)
    ).mean()
    s["com_support_error_m"] = (
      self.ep_com_err[env_ids].sum() / self.ep_com_err_steps[env_ids].sum().clamp_min(1.0)
    )
    s["final_stance_width_m"] = self.stance_width[env_ids].mean()
    s["final_foot_gap_m"] = self.foot_gap[env_ids].mean()
    s["actuated_peak_joint_vel"] = self.actuated_peak_vel[env_ids].mean()
    s["actuated_velocity_violation"] = (
      self.actuated_peak_vel[env_ids] > self.cfg.safe_recovery.evaluation_ceiling_rad_s
    ).float().mean()

  def reset_idx(self, env_ids):
    super().reset_idx(env_ids)
    if len(env_ids):
      self.com_prev_valid[env_ids] = False
      self.knee_time[env_ids] = 0
      for buf in (self.t_feet, self.t_com, self.t_rise):
        buf[env_ids] = -1.0
      self.rise_supported[env_ids] = False
      self.prev_height[env_ids] = 0.0
      self.leg_contact_sensor.reset(env_ids)
      for buf in (self.ep_knee_time, self.ep_leg_contact_time, self.ep_min_foot_share, self.ep_reverse_standing, self.ep_standing_steps,
                  self.ep_com_err, self.ep_com_err_steps, self.actuated_peak_vel):
        buf[env_ids] = 0.0
