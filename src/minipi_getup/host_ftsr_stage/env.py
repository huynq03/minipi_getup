"""HoST subclass: identical plant/step/actor/PPO, isolated reward and monitors."""

import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

import minipi_getup.host.env as baseline
from minipi_getup.host.clpai_asset import use_clpai_asset
from minipi_getup.host.train_clpai import REAL_TORQUE

from .geometry import calibrate
from .rewards import StageLogic


class StageEnv(baseline.LeggedRobot_Pi):
  def __init__(self, cfg, num_envs, device="cuda:0"):
    # Scope factory/asset/torque overrides to construction, preserving baseline
    # construction even if a caller later creates a vanilla HoST env in process.
    saved = (baseline.ASSET_XML, baseline.get_robot_cfg, baseline.TORQUE_LIMITS)
    try:
      use_clpai_asset()
      baseline.TORQUE_LIMITS = dict(REAL_TORQUE)
      super().__init__(cfg, num_envs, device)
    finally:
      baseline.ASSET_XML, baseline.get_robot_cfg, baseline.TORQUE_LIMITS = saved
    self.geometry = calibrate()
    assert self.dof_names == self.geometry["joint_names"]
    self.stage_logic = StageLogic(num_envs, device, self.dt, self.geometry)
    self.stage_settings = self.stage_logic.cfg
    self.support_poses = torch.tensor(
      self.geometry["support_poses"], device=device, dtype=torch.float
    )
    self.nominal = torch.tensor(self.geometry["nominal_pose"], device=device)
    self.physical_limits = self.robot.data.joint_pos_limits[0].clone()
    self.stage_foot_ids = torch.tensor(
      [self.body_names.index(s + "_ankle_roll_link") for s in ("l", "r")], device=device
    )
    self.site_ids = [list(self.robot.site_names).index(s + "_foot") for s in ("l", "r")]
    sensor = self.scene["stage_feet"]
    self.sensor_order = [
      next(
        i
        for i, n in enumerate(sensor.primary_names)
        if n.endswith(s + "_ankle_roll_link")
      )
      for s in ("l", "r")
    ]
    self.sat_count = torch.zeros(num_envs, 12, device=device)
    self.sample_count = torch.zeros(num_envs, device=device)
    self.max_overshoot = torch.zeros(num_envs, device=device)
    self.q_min = torch.full_like(self.dof_pos, float("inf"))
    self.q_max = -self.q_min.clone()
    self.features = {}

  def create_sim(self):
    # Same construction as host.env.create_sim, plus a measurement-only sensor.
    sensor = ContactSensorCfg(
      name="stage_feet",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    )
    self.scene = Scene(
      SceneCfg(
        num_envs=self.num_envs,
        env_spacing=self.cfg.env.env_spacing,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": baseline.get_robot_cfg(self.limit_solref)},
        sensors=(sensor,),
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
    self._create_envs()

  def measure(self):
    self.scene["stage_feet"].update(self.dt)
    force = self.scene["stage_feet"].data.force[:, self.sensor_order, 2].abs()
    bodies = self.robot.indexing.body_ids.long()
    mass = self.sim.model.body_mass[:, bodies]
    weight = mass.sum(-1) * abs(self.cfg.sim.gravity[2])
    foot_pos = self.robot.data.site_pos_w[:, self.site_ids]
    foot_quat = self.robot.data.body_link_quat_w[:, self.stage_foot_ids]
    g = self.gravity_vec[:, None].expand(-1, 2, -1)
    foot_up = -quat_apply_inverse(foot_quat.reshape(-1, 4), g.reshape(-1, 3))[
      :, 2
    ].reshape(-1, 2)
    rel = foot_pos - self.base_pos[:, None]
    local = quat_apply_inverse(
      self.base_quat[:, None].expand(-1, 2, -1).reshape(-1, 4), rel.reshape(-1, 3)
    ).reshape(-1, 2, 3)
    desired = torch.tensor(self.geometry["foot_xy"], device=self.device)
    placement = torch.exp(-((local[:, :, :2] - desired) ** 2).mean((1, 2)) / 0.06**2)
    # COM distance to the segment joining sole centers, with a conservative
    # 20mm footprint radius. No unsupported flight can satisfy support.
    com = (self.robot.data.body_com_pos_w * mass[:, :, None]).sum(1) / mass.sum(-1)[
      :, None
    ]
    a, b = foot_pos[:, 0, :2], foot_pos[:, 1, :2]
    v = b - a
    t = ((com[:, :2] - a) * v).sum(-1) / (v * v).sum(-1).clamp(min=1e-6)
    nearest = a + t.clamp(0, 1)[:, None] * v
    balance = (torch.norm(com[:, :2] - nearest, dim=-1) - 0.02).clamp(min=0)
    q = self.dof_pos
    lo, hi = self.physical_limits.T
    over = torch.maximum((lo - q).clamp(min=0), (q - hi).clamp(min=0)).max(-1).values
    distance = ((q[:, None] - self.support_poses[None]) ** 2).mean(-1).min(-1).values
    # Axes: pitch/knee/ankle-pitch mirror with same sign; roll/yaw opposite.
    mirror = torch.tensor([1, -1, -1, 1, 1, -1], device=self.device)
    symmetry = ((q[:, :6] * mirror - q[:, 6:]) ** 2).mean(-1).sqrt()
    f = dict(
      gravity=self.projected_gravity,
      height=self.root_states[:, 2],
      head_height=self.rigid_body_pos[:, self.head_indices, 2].squeeze(-1),
      foot_load=force / weight[:, None],
      foot_up=foot_up,
      placement=placement,
      balance_error=balance,
      support_pose=torch.exp(-distance / 0.45**2),
      nominal_pose=torch.exp(-((q - self.nominal) ** 2).mean(-1) / 0.3**2),
      pose_max=(q - self.nominal).abs().max(-1).values,
      symmetry_rms=symmetry,
      overshoot=over,
      ang_speed=self.base_ang_vel.norm(dim=-1),
      lin_speed=self.base_lin_vel.norm(dim=-1),
      offaxis_speed=self.base_ang_vel[:, [0, 2]].norm(dim=-1),
    )
    return f

  def compute_reward(self):
    self.features = self.measure()
    active = self.real_episode_length_buf > self.unactuated_time
    self.stage_logic.update(self.features, active)
    super().compute_reward()
    # Stage 1 exploration: reduce only motion regularization. Original
    # joint-position-limit penalty remains fully active at every stage.
    gate = self.stage_logic.last["regularization_gate"]
    for name in self.constraint_names:
      if name.startswith("regu_") and name != "regu_dof_pos_limits":
        old = self.term_values[name]
        new = old * gate
        delta = new - old
        self.rew_buf[:, self.reward_groups.index("regu")] += delta
        self.episode_sums[name] += delta
        self.term_values[name] = new
    for i, group in enumerate(self.reward_groups):
      self.episode_sums[group] = self.rew_buf[:, i]

  def _reward_stage_shaping(self):
    return self.stage_logic.last["style"]

  def _reward_stage_standing(self):
    return self.stage_logic.last["target"]

  def _substep_monitor(self):
    super()._substep_monitor()
    if hasattr(self, "sample_count"):
      active = self.real_episode_length_buf > self.unactuated_time
      self.sat_count.add_(
        (self.torques.abs() >= 0.995 * self.torque_limits).float() * active[:, None]
      )
      self.sample_count.add_(active.float())
      lo, hi = self.physical_limits.T
      over = (
        torch.maximum(
          (lo - self.dof_pos).clamp(min=0), (self.dof_pos - hi).clamp(min=0)
        )
        .max(-1)
        .values
      )
      self.max_overshoot.copy_(torch.maximum(self.max_overshoot, over * active))
      self.q_min.copy_(
        torch.where(
          active[:, None], torch.minimum(self.q_min, self.dof_pos), self.q_min
        )
      )
      self.q_max.copy_(
        torch.where(
          active[:, None], torch.maximum(self.q_max, self.dof_pos), self.q_max
        )
      )

  def _log_reset_metrics(self, ids):
    super()._log_reset_metrics(ids)
    if not hasattr(self, "stage_logic"):
      return
    logic = self.stage_logic
    s = self.reset_stats
    stable = logic.hold[ids] >= logic.cfg.hold_s
    physical = self.max_overshoot[ids] <= logic.cfg.joint_overshoot_tolerance
    # Canonical success in this task means anatomical standing held for 1 s.
    # Retain the old monitor explicitly rather than calling its height-only
    # recovery criterion the hybrid's success.
    s["legacy_success"] = s["success"]
    s["success"] = stable.float().mean()
    s.update(
      stage_upright_any_method=(logic.upright_hold[ids] >= logic.cfg.hold_s)
      .float()
      .mean(),
      stage_stable_success=stable.float().mean(),
      stage_ever_stable=logic.ever_stable[ids].float().mean(),
      stage_direct_success=(stable & ~logic.prone_seen[ids] & ~logic.side_seen[ids])
      .float()
      .mean(),
      stage_physical_success=(stable & physical).float().mean(),
      stage_prone_fraction=logic.prone_seen[ids].float().mean(),
      stage_roll_events=logic.roll_events[ids].mean(),
      stage_max_side_deg=logic.max_side[ids].mean() * 180 / torch.pi,
      stage_transitions=logic.transitions[ids].mean(),
      stage_regressions=logic.regressions[ids].mean(),
      stage_limit_overshoot=self.max_overshoot[ids].max(),
      stage_torque_saturation=(
        self.sat_count[ids].sum(-1) / (12 * self.sample_count[ids].clamp(min=1))
      ).mean(),
    )
    for i in range(3):
      s[f"stage_{i + 1}_seconds"] = logic.time_in_stage[ids, i].mean()
    for i, n in enumerate(self.dof_names):
      s[f"saturation_{n}"] = (
        self.sat_count[ids, i] / self.sample_count[ids].clamp(min=1)
      ).mean()
      valid = self.sample_count[ids] > 0
      if valid.any():
        s[f"angle_min_{n}"] = self.q_min[ids][valid, i].min()
        s[f"angle_max_{n}"] = self.q_max[ids][valid, i].max()
    for k, v in s.items():
      if k.startswith("stage_"):
        self.extras["episode"][k] = v

  def reset_idx(self, ids):
    super().reset_idx(ids)
    if len(ids) and hasattr(self, "stage_logic"):
      self.stage_logic.reset(ids)
      self.scene["stage_feet"].reset(ids)
      self.sat_count[ids] = 0
      self.sample_count[ids] = 0
      self.max_overshoot[ids] = 0
      self.q_min[ids] = float("inf")
      self.q_max[ids] = -float("inf")
