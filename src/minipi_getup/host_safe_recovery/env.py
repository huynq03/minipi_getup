"""Isolated environment delta for arbitrary-fall HoST safe recovery."""

from __future__ import annotations

import math

import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

import minipi_getup.host.env as host_env
from minipi_getup.host.env import LeggedRobot_Pi, torch_rand_float
from minipi_getup.host_safe_recovery.asset import REAL_TORQUE

RESET_CLASSES = ("supine", "prone", "left", "right", "intermediate")
FEET_CONTACT_SENSOR = "safe_recovery_feet_ground"
NONFOOT_CONTACT_SENSOR = "safe_recovery_nonfoot_ground"
SOLE_SLIP_SENSOR = "safe_recovery_sole_slip"


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


def _sole_contact_coverage(force_history: torch.Tensor, scale_n: torch.Tensor) -> torch.Tensor:
  """Fraction of sole strips contacting terrain, averaged over physics samples.

  A proxy for coverage, not an exact contact area. Saturate each strip before
  averaging so one loaded edge or one impact cannot replace broad contact.
  """
  normal = force_history[..., 2].abs()
  return (-torch.expm1(-normal / scale_n[:, None, None].clamp_min(1e-6))).mean(dim=(1, 2))


def _knee_sole_progress(knee_delta, coverage_delta, scale_rad):
  """Only simultaneous flexion and better same-foot contact earn credit."""
  return (torch.relu(knee_delta) / scale_rad).clamp(max=1.0) * torch.relu(coverage_delta)


def _com_ascent_cost(velocity, threshold, excess_scale):
  excess = torch.relu(velocity - threshold)
  return excess.square() + excess.pow(4) / excess_scale**2


def _rotation_speed_cost(angular_velocity, threshold):
  return torch.relu(torch.linalg.vector_norm(angular_velocity, dim=-1) - threshold).square()


def _contact_tangential_speed_cost(linear, angular, offset, threshold, scale):
  """Linear excess contact-point speed: dragging a loaded sole at 0.1-0.3 m/s is
  costly, unlike a squared m/s cost that only grows above ~0.3 m/s."""
  velocity = linear + torch.cross(angular, offset, dim=-1)
  return torch.relu(torch.linalg.vector_norm(velocity[..., :2], dim=-1) - threshold) / scale


def _loaded_hip_roll_excess(outward, support, soft_rad):
  """Wide outward hip roll is free on an unloaded leg (stepping) but costly while
  that leg's sole carries load, where closing it drags the sole across the ground."""
  return (torch.relu(outward - soft_rad).square() * support).mean(dim=1)


def _loaded_coverage_deficit(coverage, support, minimum):
  """Squared relative deficit, independently weighted by same-foot load."""
  return ((torch.relu(minimum - coverage) / minimum).square() * support).mean(dim=1)


def _sole_tilt_cost(foot_quat: torch.Tensor, sigma_rad: float) -> torch.Tensor:
  """Bounded tilt of the sole plane against the horizontal ground, in [0, 1).

  Covers both roll (side edge) and pitch (heel/toe), which strip coverage
  cannot distinguish because the strips run heel-to-toe.
  """
  down = torch.zeros_like(foot_quat[..., :3])
  down[..., 2] = -1.0
  gravity = quat_apply_inverse(foot_quat.reshape(-1, 4), down.reshape(-1, 3)).reshape(down.shape)
  tilt = torch.acos((-gravity[..., 2]).clamp(-1.0, 1.0))
  return -torch.expm1(-(tilt / sigma_rad).square())


def _loaded_knee_flexion(calf, support, gate, scale_rad):
  """Saturating credit for knee flexion (positive calf) on load-bearing legs."""
  flexion = -torch.expm1(-torch.relu(calf) / scale_rad)
  return gate * (flexion * support).mean(dim=1)


def _loaded_knee_hyperextension(calf, support, gate, soft_rad, span_rad):
  """Squared cost of pushing a load-bearing knee into its negative (reverse) range."""
  excess = (torch.relu(-calf - soft_rad) / span_rad).square()
  return gate * (excess * support).mean(dim=1)


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
        mode="geom", pattern=r"^[rl]_(calf|hip|ankle_pitch|ankle_roll)_collision$", entity="robot"
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
        sensors=(foot_sensor, nonfoot_sensor, ContactSensorCfg(
          name=SOLE_SLIP_SENSOR,
          primary=ContactMatch(mode="geom", pattern=r"^[rl]_foot[1-5]_collision$", entity="robot"),
          secondary=ContactMatch(mode="body", pattern="terrain"),
          fields=("found", "force", "pos", "normal", "tangent"),
          reduce="maxforce", global_frame=True,
        )),
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
    self.sole_slip_sensor: ContactSensor = self.scene[SOLE_SLIP_SENSOR]
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
    self.penalty_success_ema = torch.zeros((), device=self.device)
    self.penalty_stand_hold = torch.zeros(self.num_envs, device=self.device)
    self.penalty_stood_on_feet = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.penalty_stand_window = torch.zeros(self.num_envs, round(self.cfg.safe_recovery.penalty_stand_hold_s / self.dt), device=self.device)
    self.knee_contact_indices = torch.tensor(
      [self.dof_names.index(f"{side}_calf_joint") for side in ("l", "r")],
      device=self.device,
    )
    self.previous_contact_knees = torch.zeros(self.num_envs, 2, device=self.device)
    self.previous_sole_coverage = torch.zeros(self.num_envs, 2, device=self.device)
    self.knee_contact_progress_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.previous_contact_hip_roll = torch.zeros(self.num_envs, 2, device=self.device)
    self.previous_hip_sole_coverage = torch.zeros(self.num_envs, 2, device=self.device)
    self.hip_contact_progress_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.overspeed_peak_by_joint = torch.zeros_like(self.dof_vel)
    self.overspeed_steps_by_joint = torch.zeros_like(self.dof_vel)
    sigma, weight = [], []
    for name in self.dof_names:
      kind = next(k for k in self.cfg.safe_recovery.pose_sigma if name.endswith(f"_{k}_joint"))
      sigma.append(self.cfg.safe_recovery.pose_sigma[kind])
      weight.append(self.cfg.safe_recovery.pose_weight[kind])
    self.pose_sigma = torch.tensor(sigma, device=self.device).unsqueeze(0)
    self.pose_weight = torch.tensor(weight, device=self.device).unsqueeze(0)
    self.recovery_hip_roll_indices = torch.tensor(
      [self.dof_names.index("l_hip_roll_joint"), self.dof_names.index("r_hip_roll_joint")],
      device=self.device, dtype=torch.long,
    )
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
    self.knee_force_indices = torch.tensor(
      [i for i, name in enumerate(nonfoot_names) if name in ("l_calf_collision", "r_calf_collision")],
      dtype=torch.long,
      device=self.device,
    )
    if len(self.knee_force_indices) != 2:
      raise RuntimeError(f"unexpected knee contact geoms: {nonfoot_names}")
    if len(self.left_sole_force_indices) != 5 or len(self.right_sole_force_indices) != 5:
      raise RuntimeError(f"unexpected sole contact geoms: {foot_names}")
    if len(self.nonfoot_force_indices) != 8:
      raise RuntimeError(f"unexpected knee/calf/hip contact geoms: {nonfoot_names}")
    self.ep_left_foot_force = torch.zeros(self.num_envs, device=self.device)
    self.ep_right_foot_force = torch.zeros(self.num_envs, device=self.device)
    self.ep_nonfoot_force = torch.zeros(self.num_envs, device=self.device)
    self.ep_both_foot_support = torch.zeros(self.num_envs, device=self.device)
    self.support_memory = torch.zeros(self.num_envs, 2, device=self.device)
    self.ascent_cost_history = torch.zeros(
      self.num_envs, self.cfg.control.decimation, device=self.device
    )
    self.rotation_cost_history = torch.zeros_like(self.ascent_cost_history)
    self.slip_cost_history = torch.zeros_like(self.ascent_cost_history)
    geom_names = [name.split('/')[-1] for name in self.robot.geom_names]
    slip_names = self.sole_slip_sensor.primary_names
    self.slip_geom_indices = torch.tensor([geom_names.index(n.split('/')[-1]) for n in slip_names], device=self.device)
    self.slip_side_indices = torch.tensor([0 if n.split('/')[-1].startswith('l_') else 1 for n in slip_names], device=self.device)
    body_names = list(self.robot.body_names)
    self.sole_body_indices = torch.tensor(
      [body_names.index(f"{side}_ankle_roll_link") for side in ("l", "r")],
      dtype=torch.long, device=self.device,
    )

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
    stable_left = left * self.support_memory[:, 0]
    stable_right = right * self.support_memory[:, 1]
    return 0.5 * (left + right) + 0.5 * stable_left * stable_right

  def _sole_coverage(self):
    history = self.feet_ground_sensor.data.force_history
    if history is None:
      raise RuntimeError("sole contact force history is unavailable")
    body_ids = self.robot.indexing.body_ids.long()
    weight = self.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(self.cfg.sim.gravity[2])
    scale = weight * self.cfg.safe_recovery.sole_contact_scale_weight_fraction
    left = _sole_contact_coverage(history[:, self.left_sole_force_indices], scale)
    right = _sole_contact_coverage(history[:, self.right_sole_force_indices], scale)
    return torch.stack((left, right), dim=1)

  def _reward_sole_contact_coverage(self):
    left, right = self._sole_coverage().unbind(dim=1)
    # Active from the fallen pose: sequential placement gets credit, both soles
    # get extra credit. No height/hip/knee target or mandatory ordering gate.
    return _foot_support_score(left, right)

  def _reward_knee_sole_progress(self):
    coverage = self._sole_coverage()
    knees = self.dof_pos[:, self.knee_contact_indices]
    progress = _knee_sole_progress(
      knees - self.previous_contact_knees,
      coverage - self.previous_sole_coverage,
      self.cfg.safe_recovery.knee_contact_progress_scale_rad,
    ).mean(dim=1)
    reward = progress * self.knee_contact_progress_valid * (1.0 - self._standing_gate())
    self.previous_contact_knees.copy_(knees)
    self.previous_sole_coverage.copy_(coverage)
    self.knee_contact_progress_valid.fill_(True)
    return reward

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
    # Retain immediate credit while reserving most of the bonus for sustained load.
    stable_left = left * self.support_memory[:, 0]
    stable_right = right * self.support_memory[:, 1]
    score = 0.2 * _foot_support_score(left, right)
    score += 0.8 * _foot_support_score(stable_left, stable_right)
    return height * score

  def _reward_knee_load_support(self):
    # The calf geom is the local knee housing in the current XML. Filtered
    # terrain contacts exclude torso contact and all other support geoms.
    history = self.nonfoot_ground_sensor.data.force_history
    normal = history[:, self.knee_force_indices, :, 2].abs().mean(dim=2).sum(dim=1)
    cfg = self.cfg.safe_recovery
    body_ids = self.robot.indexing.body_ids.long()
    weight = self.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(self.cfg.sim.gravity[2])
    load = _saturating_support(
      normal, weight, cfg.contact_force_deadband_weight_fraction,
      cfg.contact_force_scale_weight_fraction,
    )
    elevated = torch.sigmoid(
      (self.root_states[:, 2] - cfg.knee_support_height_center)
      / cfg.knee_support_height_width
    )
    lifting = -torch.expm1(
      -torch.relu(self.root_states[:, 9]) / cfg.knee_lift_velocity_scale_m_s
    )
    # No upright requirement: lifting a tilted torso on knees also incurs cost.
    # Low, stationary rolling contacts retain only a small smooth penalty.
    gate = elevated + (1.0 - elevated) * lifting
    return load * gate

  def _reward_recovery_hip_roll(self):
    cfg = self.cfg.safe_recovery
    # Positive left / negative right roll abducts both legs in this robot.
    outward = self.dof_pos[:, self.recovery_hip_roll_indices] * torch.tensor(
      [1.0, -1.0], device=self.device
    )
    coverage = self._sole_coverage()
    # Reward useful outward movement, not a large held angle. Separate history
    # from the knee term makes evaluation independent of reward iteration order.
    score = _knee_sole_progress(
      outward - self.previous_contact_hip_roll,
      coverage - self.previous_hip_sole_coverage,
      cfg.recovery_hip_roll_progress_scale_rad,
    ).mean(dim=1)
    reward = score * self.hip_contact_progress_valid * (1.0 - self._standing_gate())
    self.previous_contact_hip_roll.copy_(outward)
    self.previous_hip_sole_coverage.copy_(coverage)
    self.hip_contact_progress_valid.fill_(True)
    return reward

  def _reward_loaded_hip_roll_excess(self):
    outward = self.dof_pos[:, self.recovery_hip_roll_indices] * torch.tensor(
      [1.0, -1.0], device=self.device
    )
    left, right, _ = self._normalized_supports()
    return _loaded_hip_roll_excess(
      outward, torch.stack((left, right), dim=1),
      self.cfg.safe_recovery.loaded_hip_roll_soft_rad,
    )

  def _knee_lift_gate(self):
    """Torso off the floor but not yet standing: the squat/lift phase."""
    cfg = self.cfg.safe_recovery
    elevated = torch.sigmoid(
      (self.root_states[:, 2] - cfg.knee_lift_gate_height_center)
      / cfg.knee_lift_gate_height_width
    )
    return elevated * (1.0 - self._standing_gate())

  def _reward_loaded_knee_flexion(self):
    left, right, _ = self._normalized_supports()
    return _loaded_knee_flexion(
      self.dof_pos[:, self.knee_contact_indices], torch.stack((left, right), dim=1),
      self._knee_lift_gate(), self.cfg.safe_recovery.knee_flexion_scale_rad,
    )

  def _reward_loaded_knee_hyperextension(self):
    cfg = self.cfg.safe_recovery
    left, right, _ = self._normalized_supports()
    return _loaded_knee_hyperextension(
      self.dof_pos[:, self.knee_contact_indices], torch.stack((left, right), dim=1),
      self._knee_lift_gate(), cfg.knee_hyperextension_soft_rad,
      cfg.knee_hyperextension_span_rad,
    )

  def _knee_height_credit(self):
    left, right, _ = self._ground_support_forces()
    history = self.nonfoot_ground_sensor.data.force_history
    knee = history[:, self.knee_force_indices, :, 2].abs().mean(dim=2).sum(dim=1)
    body_ids = self.robot.indexing.body_ids.long()
    weight = self.sim.model.body_mass[:, body_ids].sum(dim=1) * abs(self.cfg.sim.gravity[2])
    # Ignore incidental contact, and avoid unstable ratios at near-zero load.
    knee = torch.relu(knee - self.cfg.safe_recovery.contact_force_deadband_weight_fraction * weight)
    knee_share = knee / (left + right + knee + 0.05 * weight).clamp_min(1.0e-6)
    elevated = torch.sigmoid(
      (self.root_states[:, 2] - self.cfg.safe_recovery.knee_support_height_center)
      / self.cfg.safe_recovery.knee_support_height_width
    )
    floor = self.cfg.safe_recovery.knee_height_credit_floor
    return 1.0 - (1.0 - floor) * elevated * knee_share

  def _reward_head_height(self):
    # Preserve the original updates of head-height curriculum/monitor buffers.
    reward = super()._reward_head_height()
    return reward * self._knee_height_credit().unsqueeze(1)

  def _reward_target_base_height(self):
    return super()._reward_target_base_height() * self._knee_height_credit()

  def _reward_ballistic_ascent(self):
    return self.ascent_cost_history.mean(dim=1)

  def _reward_recovery_rotation(self):
    residual = self.cfg.safe_recovery.recovery_rotation_standing_residual
    gate = 1.0 - (1.0 - residual) * self._standing_gate()
    initial = self.cfg.safe_recovery.penalty_initial_rotation_fraction
    return (initial + (1.0 - initial) * self._penalty_curriculum_strength()) * gate * self.rotation_cost_history.mean(dim=1)

  def _recovery_regu_multiplier(self):
    initial = self.cfg.safe_recovery.penalty_initial_regu_fraction
    return initial + (1.0 - initial) * self._penalty_curriculum_strength()

  def _reward_action_rate(self):
    return super()._reward_action_rate() * self._recovery_regu_multiplier()

  def _reward_smoothness(self):
    return super()._reward_smoothness() * self._recovery_regu_multiplier()

  def _reward_loaded_sole_slip(self):
    return self.slip_cost_history.mean(dim=1)

  def _reward_loaded_sole_coverage_deficit(self):
    left, right, _ = self._normalized_supports()
    load = torch.stack((left, right), dim=1)
    return _loaded_coverage_deficit(
      self._sole_coverage(), load, self.cfg.safe_recovery.loaded_sole_min_coverage)

  def _reward_loaded_sole_tilt(self):
    left, right, _ = self._normalized_supports()
    load = torch.stack((left, right), dim=1)
    tilt = _sole_tilt_cost(
      self.robot.data.body_link_quat_w[:, self.sole_body_indices],
      self.cfg.safe_recovery.sole_tilt_sigma_rad,
    )
    # Load-weighted: an unloaded foot may swing tilted; a loaded one must be flat.
    return (load * tilt).mean(dim=1)

  def _standing_penalty_boost(self, maximum):
    """Success-curriculum strengthening with a smooth standing-state gate.

    Accept one load-bearing foot during transfer; require persistent support
    and upright/elevated posture before increasing the existing costs.
    """
    left, right, _ = self._normalized_supports()
    sustained = torch.stack((left, right), dim=1) * self.support_memory
    gate = self._penalty_curriculum_strength() * self._standing_gate() * sustained.max(dim=1).values
    return 1.0 + (maximum - 1.0) * gate

  def _penalty_curriculum_strength(self):
    cfg = self.cfg.safe_recovery
    fraction = ((self.penalty_success_ema - cfg.penalty_success_start)
                / (cfg.penalty_success_full - cfg.penalty_success_start)).clamp(0.0, 1.0)
    return fraction.square() * (3.0 - 2.0 * fraction)

  def update_success_monitor(self):
    super().update_success_monitor()
    left, right, _ = self._normalized_supports()
    feet = torch.stack((left, right), dim=1) * self.support_memory
    standing = ((self.root_states[:, 2] > host_env.STAND_HEIGHT)
                & (self.projected_gravity[:, 2] < -0.8)
                & (feet.min(dim=1).values > 0.25))
    # A 0.5 s window tolerates brief unloading, but sitting cannot qualify.
    self.penalty_stand_window = torch.roll(self.penalty_stand_window, -1, dims=1)
    self.penalty_stand_window[:, -1] = standing.float()
    self.penalty_stood_on_feet |= standing & (self.penalty_stand_window.mean(1) >= self.cfg.safe_recovery.penalty_stand_support_fraction)

  def _reward_joint_overspeed(self):
    excess = torch.relu(torch.abs(self.dof_vel) - self.cfg.safe_recovery.overspeed_soft_rad_s)
    excess_sq = excess.square()
    return (0.5 * excess_sq.mean(dim=1) + 0.5 * excess_sq.max(dim=1).values) * self._recovery_regu_multiplier()

  def _reward_default_posture(self):
    normalized = (self.dof_pos - self.default_dof_pos) / self.pose_sigma
    boost = self._standing_penalty_boost(self.cfg.safe_recovery.standing_posture_penalty_multiplier)
    return boost * self._standing_gate() * torch.mean(self.pose_weight * normalized.square(), dim=1)

  def _reward_stable_standing(self):
    cfg = self.cfg.safe_recovery
    motion = cfg.stable_joint_velocity_weight * torch.mean(self.dof_vel.square(), dim=1)
    motion += cfg.stable_base_angular_velocity_weight * torch.sum(
      self.base_ang_vel.square(), dim=1
    )
    boost = self._standing_penalty_boost(cfg.standing_motion_penalty_multiplier)
    return boost * self._standing_gate() * motion

  def _substep_monitor(self):
    self.feet_ground_sensor.update(self.cfg.sim.dt)
    self.nonfoot_ground_sensor.update(self.cfg.sim.dt)
    self.sole_slip_sensor.update(self.cfg.sim.dt)
    left, right, _ = self._normalized_supports()
    support = torch.stack((left, right), dim=1)
    cfg = self.cfg.safe_recovery
    tau = torch.where(
      support > self.support_memory,
      cfg.support_rise_time_s,
      cfg.support_decay_time_s,
    )
    self.support_memory += (1.0 - torch.exp(-self.cfg.sim.dt / tau)) * (
      support - self.support_memory
    )
    vz = self.robot.data.root_com_lin_vel_w[:, 2]
    # Immediate cost sampled every 5 ms; no success gate or foot exemption.
    strength = self._penalty_curriculum_strength()
    threshold = cfg.initial_upward_velocity_soft_m_s + strength * (cfg.upward_velocity_soft_m_s - cfg.initial_upward_velocity_soft_m_s)
    excess = torch.relu(vz - threshold)
    cost = excess.square() + strength * excess.pow(4) / cfg.ascent_excess_scale_m_s**2
    cost *= self.real_episode_length_buf > self.unactuated_time
    self.ascent_cost_history = torch.roll(self.ascent_cost_history, 1, dims=1)
    self.ascent_cost_history[:, 0] = cost
    rotation = _rotation_speed_cost(
      self.robot.data.root_link_ang_vel_w, cfg.recovery_rotation_soft_rad_s)
    rotation *= self.real_episode_length_buf > self.unactuated_time
    self.rotation_cost_history = torch.roll(self.rotation_cost_history, 1, dims=1)
    self.rotation_cost_history[:, 0] = rotation
    contact = self.sole_slip_sensor.data
    indices = self.slip_geom_indices
    d = self.robot.data
    slip = _contact_tangential_speed_cost(
      d.geom_lin_vel_w[:, indices], d.geom_ang_vel_w[:, indices],
      contact.pos - d.geom_pos_w[:, indices], cfg.sole_slip_soft_m_s, cfg.sole_slip_scale_m_s)
    weight = self.sim.model.body_mass[:, self.robot.indexing.body_ids.long()].sum(1) * abs(self.cfg.sim.gravity[2])
    load = _saturating_support(contact.force[..., 2].abs(), weight[:, None], 0.002, 0.05)
    # Instantaneous foot support: dragging during transient load transfer counts.
    slip_cost = (slip * load * support[:, self.slip_side_indices] * (contact.found > 0)).mean(1)
    slip_cost *= self.real_episode_length_buf > self.unactuated_time
    self.slip_cost_history = torch.roll(self.slip_cost_history, 1, dims=1)
    self.slip_cost_history[:, 0] = slip_cost
    super()._substep_monitor()
    speed = torch.abs(self.dof_vel)
    self.overspeed_peak_by_joint = torch.maximum(self.overspeed_peak_by_joint, speed)
    self.overspeed_steps_by_joint += (
      speed > self.cfg.safe_recovery.overspeed_soft_rad_s
    )

  def _log_reset_metrics(self, env_ids):
    if len(env_ids):
      success = (self.penalty_stood_on_feet[env_ids] & ~self.fell_after_stand[env_ids]).float().mean()
      alpha = -math.expm1(-len(env_ids) / self.cfg.safe_recovery.penalty_success_ema_episodes)
      self.penalty_success_ema += alpha * (success - self.penalty_success_ema)
    super()._log_reset_metrics(env_ids)
    self.reset_stats["penalty_foot_success_ema"] = self.penalty_success_ema.clone()
    self.reset_stats["penalty_curriculum_strength"] = self._penalty_curriculum_strength().clone()
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
      self.knee_contact_progress_valid[env_ids] = False
      self.hip_contact_progress_valid[env_ids] = False
      self.penalty_stand_hold[env_ids] = 0
      self.penalty_stand_window[env_ids] = 0
      self.penalty_stood_on_feet[env_ids] = False
      self.feet_ground_sensor.reset(env_ids)
      self.nonfoot_ground_sensor.reset(env_ids)
      self.overspeed_peak_by_joint[env_ids] = 0
      self.overspeed_steps_by_joint[env_ids] = 0
      self.ep_left_foot_force[env_ids] = 0
      self.ep_right_foot_force[env_ids] = 0
      self.ep_nonfoot_force[env_ids] = 0
      self.ep_both_foot_support[env_ids] = 0
      self.support_memory[env_ids] = 0
      self.ascent_cost_history[env_ids] = 0
      self.rotation_cost_history[env_ids] = 0
      self.slip_cost_history[env_ids] = 0
      self.sole_slip_sensor.reset(env_ids)

  def update_force_curriculum(self, env_ids):
    if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
      step = self.cfg.safe_recovery.force_curriculum_step_n
      self.force[env_ids] = (self.force[env_ids] - step).clamp_min(0.0)
      # Keep the original schedule down to 0.25, never raise a resumed scale.
      current = self.action_rescale[env_ids]
      self.action_rescale[env_ids] = torch.where(current > 0.25, (current - 0.02).clamp_min(0.25), current)
    cfg = self.cfg.safe_recovery
    if self.penalty_success_ema >= cfg.action_scale_success_threshold:
      # Below 0.25 only successful, feet-supported episodes qualify, without
      # external assistance. Failure pauses progression instead of forcing 0.10.
      eligible = self.penalty_stood_on_feet[env_ids] & (self.force[env_ids].reshape(-1) <= 0)
      ids = env_ids[eligible]
      self.action_rescale[ids] = (self.action_rescale[ids] - cfg.action_scale_finetune_step).clamp_min(cfg.action_scale_min)
