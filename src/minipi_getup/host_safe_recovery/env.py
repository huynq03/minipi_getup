"""Isolated environment delta for arbitrary-fall HoST safe recovery."""

from __future__ import annotations

import math

import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg

import minipi_getup.host.env as host_env
from minipi_getup.host.env import LeggedRobot_Pi, torch_rand_float
from minipi_getup.host_safe_recovery.asset import REAL_TORQUE

RESET_CLASSES = ("supine", "prone", "left", "right", "intermediate")
FEET_CONTACT_SENSOR = "safe_recovery_feet_ground"
NONFOOT_CONTACT_SENSOR = "safe_recovery_nonfoot_ground"


def _saturating_support(
  force_n: torch.Tensor,
  weight_n: torch.Tensor,
  deadband_fraction: float,
  scale_fraction: float,
) -> torch.Tensor:
  """Map ground-normal load to [0, 1) without rewarding impact magnitude."""
  load = torch.relu(force_n - deadband_fraction * weight_n)
  scale = (scale_fraction * weight_n).clamp_min(1.0e-6)
  return -torch.expm1(-load / scale)


def _foot_support_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
  """Independent single-foot credit plus an explicit two-foot support bonus."""
  return 0.5 * (left + right) + 0.5 * left * right


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
  """HoST environment with real limits and foot-supported safe recovery rewards."""

  def create_sim(self):
    """Build the unchanged HoST scene plus ground-filtered contact sensors."""
    foot_sensor = ContactSensorCfg(
      name=FEET_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom", pattern=r"^[rl]_foot[1-5]_collision$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=self.cfg.control.decimation,
    )
    nonfoot_sensor = ContactSensorCfg(
      name=NONFOOT_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom", pattern=r"^[rl]_(calf|hip)_collision$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=self.cfg.control.decimation,
    )
    self.scene = Scene(
      SceneCfg(
        num_envs=self.num_envs,
        env_spacing=self.cfg.env.env_spacing,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": host_env.get_robot_cfg(self.limit_solref)},
        sensors=(foot_sensor, nonfoot_sensor),
        extent=2.0,
      ),
      device=self.device,
    )
    self.sim_cfg = SimulationCfg(
      njmax=1200,
      nconmax=300,
      mujoco=MujocoCfg(
        timestep=self.cfg.sim.dt / self.physics_substeps,
        gravity=tuple(self.cfg.sim.gravity),
      ),
    )
    self.sim = Simulation(
      num_envs=self.num_envs,
      cfg=self.sim_cfg,
      spec=self.scene.spec,
      variant_info=self.scene.collect_variant_info(),
      device=self.device,
    )
    self.scene.initialize(
      mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data
    )
    self.robot = self.scene["robot"]
    self.feet_ground_sensor: ContactSensor = self.scene[FEET_CONTACT_SENSOR]
    self.nonfoot_ground_sensor: ContactSensor = self.scene[NONFOOT_CONTACT_SENSOR]
    self._create_envs()

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
    foot_names = self.feet_ground_sensor.primary_names
    nonfoot_names = self.nonfoot_ground_sensor.primary_names
    self.left_sole_force_indices = torch.tensor(
      [i for i, name in enumerate(foot_names) if name.startswith("l_foot")],
      dtype=torch.long,
      device=self.device,
    )
    self.right_sole_force_indices = torch.tensor(
      [i for i, name in enumerate(foot_names) if name.startswith("r_foot")],
      dtype=torch.long,
      device=self.device,
    )
    self.nonfoot_force_indices = torch.arange(
      len(nonfoot_names), dtype=torch.long, device=self.device
    )
    if len(self.left_sole_force_indices) != 5 or len(self.right_sole_force_indices) != 5:
      raise RuntimeError(f"unexpected sole contact geoms: {foot_names}")
    if len(self.nonfoot_force_indices) != 4:
      raise RuntimeError(f"unexpected knee/calf/hip contact geoms: {nonfoot_names}")
    self.ep_left_foot_force = torch.zeros(self.num_envs, device=self.device)
    self.ep_right_foot_force = torch.zeros(self.num_envs, device=self.device)
    self.ep_nonfoot_force = torch.zeros(self.num_envs, device=self.device)
    self.ep_both_foot_support = torch.zeros(self.num_envs, device=self.device)

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

  def _ground_support_forces(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean 5 ms ground-normal loads over the current 20 ms policy step."""
    foot_history = self.feet_ground_sensor.data.force_history
    nonfoot_history = self.nonfoot_ground_sensor.data.force_history
    if foot_history is None or nonfoot_history is None:
      raise RuntimeError("contact force history is unavailable")
    # Sensors are filtered to terrain. Absolute world-z is the normal load magnitude;
    # averaging history rejects one-substep impacts as a source of reward.
    foot_normal = foot_history[..., 2].abs().mean(dim=2)
    nonfoot_normal = nonfoot_history[..., 2].abs().mean(dim=2)
    left = foot_normal[:, self.left_sole_force_indices].sum(dim=1)
    right = foot_normal[:, self.right_sole_force_indices].sum(dim=1)
    nonfoot = nonfoot_normal[:, self.nonfoot_force_indices].sum(dim=1)
    return left, right, nonfoot

  def _normalized_supports(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_force, right_force, nonfoot_force = self._ground_support_forces()
    body_ids = self.robot.indexing.body_ids.long()
    weight = self.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(self.cfg.sim.gravity[2])
    cfg = self.cfg.safe_recovery
    args = (
      weight,
      cfg.contact_force_deadband_weight_fraction,
      cfg.contact_force_scale_weight_fraction,
    )
    return (
      _saturating_support(left_force, *args),
      _saturating_support(right_force, *args),
      _saturating_support(nonfoot_force, *args),
    )

  def _reward_foot_load_support(self):
    left_force, right_force, _ = self._ground_support_forces()
    left, right, _ = self._normalized_supports()
    self.ep_left_foot_force += left_force
    self.ep_right_foot_force += right_force
    self.ep_both_foot_support += (left * right).detach()
    return _foot_support_score(left, right)

  def _reward_nonfoot_support(self):
    _, _, nonfoot_force = self._ground_support_forces()
    _, _, nonfoot = self._normalized_supports()
    cfg = self.cfg.safe_recovery
    elevated = torch.sigmoid(
      (self.root_states[:, 2] - cfg.support_transfer_height_center)
      / cfg.support_transfer_height_width
    )
    upright = torch.sigmoid(
      ((-self.projected_gravity[:, 2]) - cfg.support_transfer_upright_center)
      / cfg.support_transfer_upright_width
    )
    # Elevation opens the gate; uprightness increases it smoothly. Contact force is
    # still nearly free in early fallen configurations but costly during weight transfer.
    transfer_gate = elevated * (0.2 + 0.8 * upright)
    self.ep_nonfoot_force += nonfoot_force
    return transfer_gate * nonfoot

  def _reward_foot_supported_height(self):
    left, right, _ = self._normalized_supports()
    cfg = self.cfg.safe_recovery
    height = torch.sigmoid(
      (self.root_states[:, 2] - cfg.foot_height_center) / cfg.foot_height_width
    )
    return height * _foot_support_score(left, right)

  def _reward_joint_overspeed(self):
    excess = torch.relu(torch.abs(self.dof_vel) - self.cfg.safe_recovery.overspeed_soft_rad_s)
    excess_sq = excess.square()
    return 0.5 * excess_sq.mean(dim=1) + 0.5 * excess_sq.max(dim=1).values

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
    self.feet_ground_sensor.update(self.cfg.sim.dt)
    self.nonfoot_ground_sensor.update(self.cfg.sim.dt)
    super()._substep_monitor()
    speed = torch.abs(self.dof_vel)
    self.overspeed_peak_by_joint = torch.maximum(self.overspeed_peak_by_joint, speed)
    self.overspeed_steps_by_joint += (
      speed > self.cfg.safe_recovery.overspeed_soft_rad_s
    )

  def _log_reset_metrics(self, env_ids):
    super()._log_reset_metrics(env_ids)
    steps = self.episode_length_buf[env_ids].float().clamp_min(1.0)
    self.reset_stats["left_foot_normal_force"] = (
      self.ep_left_foot_force[env_ids] / steps
    ).mean()
    self.reset_stats["right_foot_normal_force"] = (
      self.ep_right_foot_force[env_ids] / steps
    ).mean()
    self.reset_stats["nonfoot_normal_force"] = (
      self.ep_nonfoot_force[env_ids] / steps
    ).mean()
    self.reset_stats["both_foot_support"] = (
      self.ep_both_foot_support[env_ids] / steps
    ).mean()
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
      self.feet_ground_sensor.reset(env_ids)
      self.nonfoot_ground_sensor.reset(env_ids)
      self.overspeed_peak_by_joint[env_ids] = 0
      self.overspeed_steps_by_joint[env_ids] = 0
      self.ep_left_foot_force[env_ids] = 0
      self.ep_right_foot_force[env_ids] = 0
      self.ep_nonfoot_force[env_ids] = 0
      self.ep_both_foot_support[env_ids] = 0

  def update_force_curriculum(self, env_ids):
    if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
      step = self.cfg.safe_recovery.force_curriculum_step_n
      self.force[env_ids] = (self.force[env_ids] - step).clamp_min(0.0)
      self.action_rescale[env_ids] = (self.action_rescale[env_ids] - 0.02).clamp_min(0.25)
