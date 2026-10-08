"""MuJoCo/MJLab port of HoST ``LeggedRobot_Pi`` (``envs/pi/pi_host_ground.py``).

The class keeps the release's method names, buffer names, call order and formulas. Only
the simulator calls are replaced:

==========================================  ===========================================
Isaac Gym / legged_gym                      MJLab (mujoco_warp)
==========================================  ===========================================
``gym.simulate`` (5 ms PhysX TGS step)      ``Simulation.step`` (5 ms MuJoCo step)
``set_dof_actuation_force_tensor``          motor actuators, ``set_joint_effort_target``
``apply_rigid_body_force_tensors``          ``xfrc_applied`` on ``base_link`` (world
(ENV_SPACE, at COM, next simulate only)     frame, at COM, written before the next step)
``refresh_*_tensor`` after simulate         ``Simulation.forward`` + entity data reads
``root_states`` (xyzw, COM lin vel)         root link pose (wxyz), root COM lin vel
``rigid_body_states[..., 0:3]``             ``body_link_pos_w``
``dof_state``                               ``joint_pos`` / ``joint_vel``
``set_*_state_tensor_indexed``              ``write_root_state`` / ``write_joint_state``
rigid shape / body props at creation        per-world ``geom_friction``, ``body_mass``,
                                            ``body_inertia``, ``body_ipos``
==========================================  ===========================================

Quaternions are kept in MuJoCo's (w, x, y, z) order internally; every place the
release reads a quaternion it only rotates vectors with it, so the math is identical.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from minipi_getup.host.config import PiCfg, class_to_dict

ASSET_XML = Path(__file__).parent / "assets" / "pi_12dof_host.xml"

# MuJoCo integration steps per Isaac Gym ``simulate`` call (sim.dt = 5 ms). HoST's
# control loop is unchanged: torques (PD, delay buffer, DR factors) and the pull force are
# computed once per 5 ms substep and held, as PhysX holds them for one simulate. With a
# single 5 ms MuJoCo step the light, 20 N m-driven Pi links blow up under exploratory
# actions (validate/settle study: base speed > 20 m/s in ~0.25 per 1k env-steps); PhysX
# TGS (8 position iterations, 1 m/s max depenetration velocity) does not, so MuJoCo
# integrates each 5 ms in smaller steps.
PHYSICS_SUBSTEPS = 2

# URDF ``effort`` and ``velocity`` of every Pi joint (Isaac Gym dof_props).
URDF_EFFORT = 20.0
URDF_VELOCITY = 5.0
# Per-joint torque limits keyed by joint type ({"hip_pitch": 16.0, ..., "ankle_roll": 6.0},
# every joint must match exactly one key); None keeps URDF_EFFORT on every joint (the release).
TORQUE_LIMITS: dict[str, float] | None = None


def torque_limit(joint_name: str) -> float:
  if TORQUE_LIMITS is None:
    return URDF_EFFORT
  (v,) = [v for k, v in TORQUE_LIMITS.items() if joint_name.endswith(f"_{k}_joint")]
  return v


# PhysX combines a robot shape's friction with the ground plane's by averaging (default
# combine mode). The plane has static 0.8 / dynamic 0.7; MuJoCo has one coefficient, so
# the robot geoms get priority and mu = (mu_robot + 0.75) / 2.
GROUND_FRICTION_MEAN = 0.5 * (0.8 + 0.7)
# Robot shape friction used when friction randomization is off (evaluation only): the
# midpoint of the release's training range [0.1, 1].
NOMINAL_ROBOT_FRICTION = 0.55


def torch_rand_float(lower, upper, shape, device):
  return (upper - lower) * torch.rand(*shape, device=device) + lower


def sigmoid(x, value_at_1):
  """g1_utils.sigmoid (gaussian variant)."""
  scale = np.sqrt(-2 * np.log(value_at_1))
  return torch.exp(-0.5 * (x * scale) ** 2)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
  """g1_utils.tolerance, unchanged (returns float64 where the margin applies)."""
  lower, upper = bounds
  assert lower < upper
  assert margin >= 0
  in_bounds = torch.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = torch.where(in_bounds, 1.0, 0)
  else:
    d = torch.where(x < lower, lower - x, x - upper) / margin
    value = torch.where(in_bounds, 1.0, sigmoid(d.double(), value_at_margin))
  return value


def get_robot_cfg(limit_solref: tuple[float, float] | None = None) -> EntityCfg:
  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ASSET_XML))
    # Robot friction wins over the plane's (see GROUND_FRICTION_MEAN).
    for g in spec.geoms:
      if g.contype:
        g.priority = 1
    if limit_solref is not None:  # diagnostic only; default keeps MuJoCo's (0.02, 1)
      for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
          j.solref_limit = list(limit_solref)
    return spec

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.351), rot=(1.0, 0.0, 0.0, 0.0)
    ),
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=(".*_joint",),
          effort_limit=URDF_EFFORT,
          armature=PiCfg.asset.armature,
        ),
      )
      if TORQUE_LIMITS is None
      else tuple(
        BuiltinMotorActuatorCfg(
          target_names_expr=(f".*_{k}_joint",),
          effort_limit=v,
          armature=PiCfg.asset.armature,
        )
        for k, v in TORQUE_LIMITS.items()
      ),
      # ctrl order = joint order, so torques can be written as one (N, 12) block.
    ),
    sort_actuators=True,
  )


class LeggedRobot_Pi:
  def __init__(
    self,
    cfg: PiCfg,
    num_envs: int,
    device: str,
    headless: bool = True,
    physics_substeps: int | None = None,
    joint_vel_cap: float | None = None,
    limit_solref: tuple[float, float] | None = None,
  ):
    """``joint_vel_cap`` / ``limit_solref`` are diagnostics for the PhysX comparison
    (``maxJointVelocity`` clamp, stiffer joint limits); both are off by default."""
    self.cfg = cfg
    self.joint_vel_cap = joint_vel_cap
    self.limit_solref = limit_solref
    self.physics_substeps = (
      PHYSICS_SUBSTEPS if physics_substeps is None else physics_substeps
    )
    self.device = device
    self.headless = headless
    self.height_samples = None
    self.debug_viz = False
    self.init_done = False
    self._parse_cfg(self.cfg)
    self.num_real_dofs = cfg.env.num_dofs

    # BaseTask.__init__
    self.num_envs = num_envs
    self.num_obs = cfg.env.num_observations
    self.num_privileged_obs = cfg.env.num_privileged_obs
    self.num_actions = cfg.env.num_actions
    self.reward_groups = self.cfg.rewards.reward_groups
    self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=device)
    self.rew_buf = torch.zeros(
      self.num_envs, self.cfg.rewards.num_reward_groups, device=device
    )
    self.reset_buf = torch.ones(self.num_envs, device=device, dtype=torch.long)
    self.episode_length_buf = torch.zeros(
      self.num_envs, device=device, dtype=torch.long
    )
    self.real_episode_length_buf = torch.zeros(
      self.num_envs, device=device, dtype=torch.long
    )
    self.time_out_buf = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
    self.privileged_obs_buf = None
    self.extras = {}
    self.create_sim()

    self.num_one_step_obs = self.cfg.env.num_one_step_observations
    self.actor_history_length = self.cfg.env.num_actor_history
    self.actor_proprioceptive_obs_length = (
      self.num_one_step_obs * self.actor_history_length
    )

    self._init_buffers()
    self._prepare_reward_function()
    self.init_done = True
    self.unactuated_time = self.cfg.env.unactuated_timesteps
    self.unactuated_time *= 0.02 / self.dt
    self.is_gaussian = cfg.rewards.is_gaussian

  # ------------------------------------------------------------------ BaseTask API
  def get_observations(self):
    return self.obs_buf

  def get_privileged_observations(self):
    return self.privileged_obs_buf

  def reset(self):
    """Reset all robots."""
    self.reset_idx(torch.arange(self.num_envs, device=self.device))
    obs, privileged_obs, _, _, _ = self.step(
      torch.zeros(self.num_envs, self.num_actions, device=self.device)
    )
    return obs, privileged_obs

  # ------------------------------------------------------------------ step
  def step(self, actions):
    clip_actions = self.cfg.normalization.clip_actions
    self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

    self.peak_torque_step = torch.zeros_like(self.torques)  # PORT: monitoring
    self.step_energy = torch.zeros(self.num_envs, device=self.device)
    for _ in range(self.cfg.control.decimation):
      self.actions *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
      self.torques = self._compute_torques(self.actions).view(self.torques.shape)
      self.robot.set_joint_effort_target(self.torques)
      self.robot.write_data_to_sim()
      # Isaac Gym applies a force tensor during the *next* simulate call; the release
      # sets it after each simulate, so each step uses the force computed after the
      # previous one (including across a reset, as in the release).
      self.robot.data.write_external_wrench(
        self.pending_force, None, body_ids=self.base_indices
      )
      # One Isaac 5 ms simulate == PHYSICS_SUBSTEPS MuJoCo steps with the torque and
      # the external force held (see PHYSICS_SUBSTEPS).
      for _ in range(self.physics_substeps):
        self.sim.step()
        if self.joint_vel_cap is not None:
          qd = self.robot.data.joint_vel
          self.robot.write_joint_velocity_to_sim(
            qd.clamp(-self.joint_vel_cap, self.joint_vel_cap)
          )
      self._refresh_dof_state_tensor()

      # vertical pulling force (world +Z, at the base_link COM)
      if self.cfg.curriculum.pull_force:
        force_tensor = torch.zeros(
          [self.num_envs, len(self.base_indices), 3], device=self.device
        )
        force_tensor[:, :, 2] = self.force
        force_tensor *= (
          self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        ).unsqueeze(1)
        if not self.cfg.curriculum.no_orientation:
          force_tensor *= (
            (self.projected_gravity[:, 2] < -0.8).unsqueeze(1).unsqueeze(1)
          )
        self.pending_force = force_tensor
      else:
        self.pending_force = torch.zeros_like(self.pending_force)
      self._substep_monitor()
    self.post_physics_step()

    clip_obs = self.cfg.normalization.clip_observations
    self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
    return (
      self.obs_buf,
      self.privileged_obs_buf,
      self.rew_buf,
      self.reset_buf,
      self.extras,
    )

  def post_physics_step(self):
    # refresh_actor_root_state / rigid_body_state: MuJoCo's mj_step leaves kinematics
    # one substep stale, Isaac's refresh does not.
    self.sim.forward()
    self._refresh_root_and_body_states()

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.real_episode_length_buf += 1

    self.base_pos[:] = self.root_states[:, 0:3]
    self.base_quat[:] = self.root_states[:, 3:7]
    self.base_lin_vel[:] = quat_apply_inverse(self.base_quat, self.root_states[:, 7:10])
    self.base_ang_vel[:] = quat_apply_inverse(
      self.base_quat, self.root_states[:, 10:13]
    )
    self.projected_gravity[:] = quat_apply_inverse(self.base_quat, self.gravity_vec)

    self.check_termination()
    self.compute_reward()
    self.update_success_monitor()
    env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    self.reset_idx(env_ids)
    self.compute_observations()

    self.last_last_actions[:] = self.last_actions[:]
    self.last_actions[:] = self.actions[:]
    self.last_dof_vel[:] = self.dof_vel[:]
    self.last_root_vel[:] = self.root_states[:, 7:13]
    self.last_last_dof_pos[:] = self.last_dof_pos[:]
    self.last_dof_pos[:] = self.dof_pos[:]

  def check_termination(self):
    # terminate_after_contacts_on = [] -> no contact termination.
    self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.time_out_buf = self.episode_length_buf > self.max_episode_length
    self.reset_buf |= self.time_out_buf

    self.dof_vel_out = (
      torch.abs(self.dof_vel.max(dim=1).values) > self.cfg.curriculum.dof_vel_limit
    ) & (self.real_episode_length_buf > self.unactuated_time)
    self.reset_buf |= self.dof_vel_out

    self.base_vel_out = (
      torch.norm(self.base_lin_vel[:, :3], dim=-1) > self.cfg.curriculum.base_vel_limit
    ) & (self.real_episode_length_buf > self.unactuated_time)
    self.reset_buf |= self.base_vel_out

  def reset_idx(self, env_ids):
    if len(env_ids) == 0:
      return

    self.extras["episode"] = {}
    self.extras["episode"]["base_height"] = self.old_headheight[env_ids].mean()
    self._log_reset_metrics(env_ids)

    self._reset_dofs(env_ids)
    self._reset_root_states(env_ids)

    self.update_force_curriculum(env_ids)

    self.last_actions[env_ids] = 0.0
    self.last_last_actions[env_ids] = 0.0
    self.last_last_dof_pos[env_ids] = 0
    self.last_dof_pos[env_ids] = 0
    self.last_dof_vel[env_ids] = 0.0
    self.episode_length_buf[env_ids] = 0
    self.real_episode_length_buf[env_ids] = 0
    self.reset_buf[env_ids] = 1
    self.old_headheight[env_ids] = 0
    self.max_headheight[env_ids] = 0
    self.delay_buffer[:, env_ids, :] = 0.0

    for key in self.episode_sums.keys():
      self.extras["episode"]["rew_" + key] = (
        torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
      )
      self.episode_sums[key][env_ids] = 0.0
    if self.cfg.env.send_timeouts:
      self.extras["time_outs"] = self.time_out_buf

    self.extras["episode"]["force"] = self.force.mean()
    self.extras["episode"]["action_scale"] = self.action_rescale

    dr = self.cfg.domain_rand
    n = len(env_ids)
    if dr.randomize_kp:
      self.Kp_factors[env_ids] = torch_rand_float(
        dr.kp_range[0], dr.kp_range[1], (n, self.num_dofs), self.device
      )
    if dr.randomize_kd:
      self.Kd_factors[env_ids] = torch_rand_float(
        dr.kd_range[0], dr.kd_range[1], (n, self.num_dofs), self.device
      )
    if dr.randomize_actuation_offset:
      self.actuation_offset[env_ids] = torch_rand_float(
        dr.actuation_offset_range[0],
        dr.actuation_offset_range[1],
        (n, self.num_dof),
        self.device,
      ) * self.torque_limits.unsqueeze(0)
    if dr.randomize_motor_strength:
      self.motor_strength[env_ids] = torch_rand_float(
        dr.motor_strength_range[0],
        dr.motor_strength_range[1],
        (n, self.num_dof),
        self.device,
      )
    if dr.delay:
      self.delay_idx[env_ids] = torch.randint(
        low=0, high=dr.max_delay_timesteps, size=(n,), device=self.device
      )

  def compute_reward(self):
    if not self.is_gaussian:
      raise NotImplementedError
    self.rew_buf[:, :] = 0
    task_group_index = self.reward_groups.index("task")
    self.rew_buf[:, task_group_index] = 1
    self.term_values = {}
    for i in range(len(self.reward_functions)):
      name = self.reward_names[i]
      rew = self.reward_functions[i]() * self.reward_scales[name]
      if len(rew.shape) == 2 and rew.shape[1] == 1:
        rew = rew.squeeze(1)
      self.rew_buf[:, task_group_index] *= rew  # follow "Learning to Get Up"
      self.episode_sums[name] += rew
      self.term_values[name] = rew

    for i in range(len(self.constraints)):
      name = self.constraint_names[i]
      reward_group_name = name.split("_")[0]
      rew = self.constraints[i]() * self.constraints_scales[name]
      task_group_index = self.reward_groups.index(reward_group_name)
      self.rew_buf[:, task_group_index] += rew
      self.episode_sums[name] += rew
      self.term_values[name] = rew
      if self.cfg.constraints.only_positive_rewards:
        self.rew_buf[:, task_group_index] = torch.clip(
          self.rew_buf[:, task_group_index], min=0.0
        )

    if "termination" in self.constraints_scales:
      raise NotImplementedError  # not active for pi_ground

    for rg in self.reward_groups:
      idx = self.reward_groups.index(rg)
      self.episode_sums[rg] = self.rew_buf[:, idx]

  def compute_observations(self):
    current_obs = torch.cat(
      (
        self.base_ang_vel * self.obs_scales.ang_vel,
        self.projected_gravity,
        self.dof_pos * self.obs_scales.dof_pos,
        self.dof_vel * self.obs_scales.dof_vel,
        self.actions,
        self.action_rescale + (torch.rand_like(self.action_rescale) - 0.5) * 0.05,
      ),
      dim=-1,
    )
    if self.add_noise:
      current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec
    current_obs *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
    self.obs_buf = torch.cat(
      (
        self.obs_buf[:, self.num_one_step_obs : self.actor_proprioceptive_obs_length],
        current_obs,
      ),
      dim=-1,
    )

  # ------------------------------------------------------------------ sim creation
  def create_sim(self):
    self.scene = Scene(
      SceneCfg(
        num_envs=self.num_envs,
        env_spacing=self.cfg.env.env_spacing,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_robot_cfg(self.limit_solref)},
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

  def _create_envs(self):
    robot = self.robot
    body_names = list(robot.body_names)
    self.body_names = body_names
    self.dof_names = list(robot.joint_names)
    self.num_dof = len(self.dof_names)
    self.num_dofs = len(self.dof_names)
    self.num_bodies = len(body_names)
    ids = lambda names: torch.tensor(  # noqa: E731
      [body_names.index(n) for n in names], dtype=torch.long, device=self.device
    )
    jids = lambda names: torch.tensor(  # noqa: E731
      [self.dof_names.index(n) for n in names], dtype=torch.long, device=self.device
    )
    a = self.cfg.asset

    # _process_dof_props: URDF limits; soft limits multiply both ends.
    lim = robot.data.joint_pos_limits[0].clone()
    self.dof_pos_limits = lim * self.cfg.rewards.soft_dof_pos_limit
    self.dof_vel_limits = torch.full((self.num_dof,), URDF_VELOCITY, device=self.device)
    self.torque_limits = torch.tensor(
      [torque_limit(n) for n in self.dof_names], device=self.device
    )

    feet_names = [s for s in body_names if a.foot_name in s and "auxiliary" not in s]
    self.feet_indices = ids(feet_names)
    self.left_foot_indices = ids(
      [
        s
        for s in body_names
        if a.left_foot_name in s and "keyframe" not in s and "auxiliary" not in s
      ]
    )
    self.right_foot_indices = ids(
      [
        s
        for s in body_names
        if a.right_foot_name in s and "keyframe" not in s and "auxiliary" not in s
      ]
    )
    self.base_indices = [body_names.index(s) for s in body_names if a.base_name in s]
    self.torso_link_index = body_names.index("base_link")
    self.head_names = [s for s in body_names if a.head_name in s]
    self.head_indices = ids(self.head_names)
    self.left_knee_indices = ids(
      [s for s in body_names if a.left_knee_name in s and "keyframe" not in s]
    )
    self.right_knee_indices = ids(
      [s for s in body_names if a.right_knee_name in s and "keyframe" not in s]
    )
    self.left_ankle_names = [
      s
      for t in a.left_ankle_names
      for s in body_names
      if t in s and "keyframe" not in s
    ]
    self.right_ankle_names = [
      s
      for t in a.right_ankle_names
      for s in body_names
      if t in s and "keyframe" not in s
    ]
    self.left_ankle_indices = ids(self.left_ankle_names)
    self.right_ankle_indices = ids(self.right_ankle_names)

    self.knee_joint_indices = jids(a.knee_joints)
    self.ankle_joint_indices = jids(a.ankle_joints)
    self.hip_joint_indices = torch.cat(
      (jids(a.left_hip_joints), jids(a.right_hip_joints))
    )
    self.hip_roll_joint_indices = torch.cat(
      (jids(a.left_hip_roll_joints), jids(a.right_hip_roll_joints))
    )
    self.hip_pitch_joint_indices = torch.cat(
      (jids(a.left_hip_pitch_joints), jids(a.right_hip_pitch_joints))
    )
    self.left_leg_joints_indices = jids(a.left_leg_joints)
    self.right_leg_joints_indices = jids(a.right_leg_joints)

    pos = self.cfg.init_state.pos
    x, y, z, w = self.cfg.init_state.rot  # release order x,y,z,w
    q = torch.tensor([w, x, y, z], dtype=torch.float)
    q = q / q.norm()  # PhysX requires a unit quaternion
    self.base_init_state = torch.tensor(
      pos + q.tolist() + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel,
      dtype=torch.float,
      device=self.device,
    )
    self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
    self._process_rigid_props()

  def _process_rigid_props(self):
    """_process_rigid_shape_props + _process_rigid_body_props, once at creation."""
    dr = self.cfg.domain_rand
    model = self.sim.model
    robot_bodies = self.robot.indexing.body_ids.long()  # global ids, robot body order
    robot_geoms = self.robot.indexing.geom_ids.long()
    fields = ("geom_friction", "body_mass", "body_inertia", "body_ipos")
    self.sim.expand_model_fields(fields)
    N = self.num_envs

    col = torch.tensor(
      [int(g) for g in robot_geoms.tolist() if self.sim.mj_model.geom_contype[g]],
      device=self.device,
    )
    if dr.randomize_friction:
      self.friction_coeffs = torch_rand_float(
        dr.friction_range[0], dr.friction_range[1], (N, 1), self.device
      )
    else:
      self.friction_coeffs = torch.full(
        (N, 1), NOMINAL_ROBOT_FRICTION, device=self.device
      )
    mu = 0.5 * (self.friction_coeffs + GROUND_FRICTION_MEAN)
    model.geom_friction[:, col, 0] = mu.expand(N, len(col))
    if dr.randomize_restitution:
      # No MuJoCo counterpart (contacts are not restitution-based); sampled for parity.
      self.restitution_coeffs = torch_rand_float(
        dr.restitution_range[0], dr.restitution_range[1], (N, 1), self.device
      )

    default_mass = self.sim.get_default_field("body_mass")[robot_bodies].clone()
    default_inertia = self.sim.get_default_field("body_inertia")[robot_bodies].clone()
    default_ipos = self.sim.get_default_field("body_ipos")[robot_bodies].clone()
    self.default_rigid_body_mass = default_mass
    t = self.torso_link_index
    mass = default_mass.unsqueeze(0).repeat(N, 1)
    ipos = default_ipos.unsqueeze(0).repeat(N, 1, 1)
    if dr.randomize_payload_mass:
      self.payload = torch_rand_float(
        dr.payload_mass_range[0], dr.payload_mass_range[1], (N, 1), self.device
      )
      mass[:, t] = default_mass[t] + self.payload[:, 0]
    if dr.randomize_com_displacement:
      self.com_displacement = torch_rand_float(
        dr.com_displacement_range[0], dr.com_displacement_range[1], (N, 3), self.device
      )
      self.com_displacement[:, 0] *= 4
      self.com_displacement[:, 1] *= 4
      self.com_displacement[:, 2] *= 2
      ipos[:, t] = default_ipos[t] + self.com_displacement
    scale = torch.ones_like(mass)
    if dr.randomize_link_mass:
      # The release rescales every body, the torso included (its ``if i == torso:
      # pass`` does nothing), so this overwrites the payload above.
      scale = torch_rand_float(
        dr.link_mass_range[0], dr.link_mass_range[1], mass.shape, self.device
      )
      mass = scale * default_mass.unsqueeze(0)
    # recomputeInertia=True: keep each body's inertia consistent with its new mass.
    inertia_scale = torch.where(
      default_mass.unsqueeze(0) > 0, mass / default_mass.clamp(min=1e-12), 1.0
    )
    model.body_mass[:, robot_bodies] = mass
    model.body_inertia[:, robot_bodies] = default_inertia.unsqueeze(
      0
    ) * inertia_scale.unsqueeze(-1)
    model.body_ipos[:, robot_bodies] = ipos
    self.sim.recompute_constants(RecomputeLevel.set_const)

  # ------------------------------------------------------------------ state refresh
  def _refresh_dof_state_tensor(self):
    self.dof_pos[:] = self.robot.data.joint_pos
    self.dof_vel[:] = self.robot.data.joint_vel

  def _refresh_root_and_body_states(self):
    d = self.robot.data
    self.root_states[:, 0:3] = d.root_link_pos_w
    self.root_states[:, 3:7] = d.root_link_quat_w
    self.root_states[:, 7:10] = d.root_com_lin_vel_w  # PhysX reports COM velocity
    self.root_states[:, 10:13] = d.root_link_ang_vel_w
    self.rigid_body_pos[:] = d.body_link_pos_w
    self._refresh_dof_state_tensor()

  # ------------------------------------------------------------------ callbacks
  def _compute_torques(self, actions):
    actions_scaled = actions * self.action_rescale
    self.joint_pos_target = self.dof_pos + actions_scaled
    if self.cfg.domain_rand.delay:
      self.delay_buffer = torch.concat(
        (self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0
      )
      self.joint_pos_target = (
        self.dof_pos
        + self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
      )
    else:
      self.joint_pos_target = self.dof_pos + actions_scaled
    control_type = self.cfg.control.control_type
    if control_type == "P":
      torques = (
        self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos)
        - self.d_gains * self.Kd_factors * self.dof_vel
      )
    else:
      raise NameError(f"Unknown controller type: {control_type}")
    torques = self.motor_strength * torques + self.actuation_offset
    return torch.clip(torques, -self.torque_limits, self.torque_limits)

  def _reset_dofs(self, env_ids):
    dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
    dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
    dr = self.cfg.domain_rand
    if dr.randomize_initial_joint_pos:
      init_dos_pos = self.default_dof_pos * torch_rand_float(
        dr.initial_joint_pos_scale[0],
        dr.initial_joint_pos_scale[1],
        (len(env_ids), self.num_dof),
        self.device,
      )
      init_dos_pos += torch_rand_float(
        dr.initial_joint_pos_offset[0],
        dr.initial_joint_pos_offset[1],
        (len(env_ids), self.num_dof),
        self.device,
      )
      self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
    else:
      self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
        0.5, 1.5, (len(env_ids), self.num_dof), self.device
      )
    self.dof_vel[env_ids] = 0.0
    self.robot.write_joint_state_to_sim(
      self.dof_pos[env_ids], self.dof_vel[env_ids], env_ids=env_ids
    )

  def _reset_root_states(self, env_ids):
    self.root_states[env_ids] = self.base_init_state
    self.root_states[env_ids, :3] += self.env_origins[env_ids]
    # base_quat is a view of root_states in the release.
    self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
    self.robot.write_root_state_to_sim(self.root_states[env_ids], env_ids=env_ids)

  def update_force_curriculum(self, env_ids):
    if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
      self.force[env_ids] = (self.force[env_ids] - 20).clamp(0, np.inf)
      self.action_rescale[env_ids] = (self.action_rescale[env_ids] - 0.02).clamp(
        0.25, np.inf
      )

  def _get_noise_scale_vec(self, cfg):
    start_index = 6
    noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
    self.add_noise = self.cfg.noise.add_noise
    noise_scales = self.cfg.noise.noise_scales
    noise_level = self.cfg.noise.noise_level
    noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    noise_vec[3:6] = noise_scales.gravity * noise_level
    nd = self.num_real_dofs
    noise_vec[start_index : start_index + nd] = (
      noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    )
    noise_vec[start_index + nd : start_index + 2 * nd] = (
      noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    )
    noise_vec[start_index + 2 * nd : start_index + 3 * self.num_actions] = 0.0
    return noise_vec

  def _init_buffers(self):
    N, dev = self.num_envs, self.device
    self.root_states = torch.zeros(N, 13, device=dev)
    self.rigid_body_pos = torch.zeros(N, self.num_bodies, 3, device=dev)
    self.dof_pos = torch.zeros(N, self.num_dof, device=dev)
    self.dof_vel = torch.zeros(N, self.num_dof, device=dev)
    self.sim.forward()
    self._refresh_root_and_body_states()
    self.base_quat = self.root_states[:, 3:7]  # view, as in the release
    self.base_pos = self.root_states[:, 0:3].clone()

    self.common_step_counter = 0
    self.extras = {}
    self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
    self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat((N, 1))
    self.torques = torch.zeros(N, self.num_real_dofs, device=dev)
    self.p_gains = torch.zeros(self.num_real_dofs, device=dev)
    self.d_gains = torch.zeros(self.num_real_dofs, device=dev)
    self.actions = torch.zeros(N, self.num_actions, device=dev)
    self.last_actions = torch.zeros(N, self.num_actions, device=dev)
    self.last_last_actions = torch.zeros(N, self.num_actions, device=dev)
    self.last_dof_vel = torch.zeros_like(self.dof_vel)
    self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
    self.last_dof_pos = torch.zeros_like(self.dof_pos)
    self.last_last_dof_pos = torch.zeros_like(self.dof_pos)
    self.base_lin_vel = quat_apply_inverse(self.base_quat, self.root_states[:, 7:10])
    self.base_ang_vel = quat_apply_inverse(self.base_quat, self.root_states[:, 10:13])
    self.projected_gravity = quat_apply_inverse(self.base_quat, self.gravity_vec)
    self.old_headheight = torch.zeros(N, device=dev).unsqueeze(1)
    self.max_headheight = torch.zeros(N, device=dev).unsqueeze(1)
    self.force = self.cfg.curriculum.force * torch.ones(N, device=dev).unsqueeze(1)
    self.action_rescale = self.cfg.control.action_scale * torch.ones(
      N, device=dev
    ).unsqueeze(1)
    self.delay_buffer = torch.zeros(
      self.cfg.domain_rand.max_delay_timesteps, N, self.num_actions, device=dev
    )
    self.pending_force = torch.zeros(N, len(self.base_indices), 3, device=dev)
    self.joint_pos_target = torch.zeros(N, self.num_dof, device=dev)

    self.default_dof_pos = torch.zeros(self.num_dof, device=dev)
    self.target_dof_pos = torch.zeros(N, self.num_dof, device=dev)
    for i in range(self.num_dofs):
      name = self.dof_names[i]
      self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
      self.target_dof_pos[:, i] = self.cfg.init_state.target_joint_angles[name]
      found = False
      for dof_name in self.cfg.control.stiffness.keys():
        if dof_name in name:
          self.p_gains[i] = self.cfg.control.stiffness[dof_name]
          self.d_gains[i] = self.cfg.control.damping[dof_name]
          found = True
      if not found:
        raise ValueError(f"PD gain of joint {name} not defined")
    self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    dr = self.cfg.domain_rand
    self.Kp_factors = torch.ones(N, self.num_dofs, device=dev)
    self.Kd_factors = torch.ones(N, self.num_dofs, device=dev)
    self.actuation_offset = torch.zeros(N, self.num_dofs, device=dev)
    self.motor_strength = torch.ones(N, self.num_dofs, device=dev)
    if dr.randomize_kp:
      self.Kp_factors = torch_rand_float(
        dr.kp_range[0], dr.kp_range[1], (N, self.num_dofs), dev
      )
    if dr.randomize_kd:
      self.Kd_factors = torch_rand_float(
        dr.kd_range[0], dr.kd_range[1], (N, self.num_dofs), dev
      )
    if dr.randomize_actuation_offset:
      self.actuation_offset = torch_rand_float(
        dr.actuation_offset_range[0],
        dr.actuation_offset_range[1],
        (N, self.num_dof),
        dev,
      ) * self.torque_limits.unsqueeze(0)
    if dr.randomize_motor_strength:
      self.motor_strength = torch_rand_float(
        dr.motor_strength_range[0], dr.motor_strength_range[1], (N, self.num_dofs), dev
      )
    self.delay_idx = torch.randint(
      low=0, high=dr.max_delay_timesteps, size=(N,), device=dev
    )
    if not dr.delay:
      self.delay_idx[:] = dr.max_delay_timesteps - 1
    self._init_monitor()

  def _prepare_reward_function(self):
    for key in list(self.reward_scales.keys()):
      if self.reward_scales[key] == 0:
        self.reward_scales.pop(key)
      else:
        self.reward_scales[key] *= 1
    for key in list(self.constraints_scales.keys()):
      if self.constraints_scales[key] == 0:
        self.constraints_scales.pop(key)
      else:
        self.constraints_scales[key] *= self.dt

    self.reward_functions = []
    self.reward_names = []
    for name in self.reward_scales.keys():
      if name == "termination":
        continue
      self.reward_names.append(name)
      self.reward_functions.append(
        getattr(self, "_reward_" + "_".join(name.split("_")[1:]))
      )
    self.constraints = []
    self.constraint_names = []
    for name in self.constraints_scales.keys():
      self.constraint_names.append(name)
      self.constraints.append(getattr(self, "_reward_" + "_".join(name.split("_")[1:])))

    self.episode_sums = {
      name: torch.zeros(self.num_envs, device=self.device)
      for name in self.reward_scales.keys()
    }
    for name in self.constraint_names:
      self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device)

  def _parse_cfg(self, cfg):
    self.dt = self.cfg.control.decimation * self.cfg.sim.dt
    self.obs_scales = self.cfg.normalization.obs_scales
    self.reward_scales = class_to_dict(self.cfg.rewards.scales)
    self.constraints_scales = class_to_dict(self.cfg.constraints.scales)
    self.max_episode_length_s = self.cfg.env.episode_length_s
    self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
    self.cfg.domain_rand.push_interval = np.ceil(
      self.cfg.domain_rand.push_interval_s / self.dt
    )

  # ------------------------------------------------------------------ task rewards
  def _reward_orientation(self):
    return tolerance(
      -self.projected_gravity[:, 2],
      [self.cfg.rewards.orientation_threshold, np.inf],
      1.0,
      0.05,
    )

  def _reward_head_height(self):
    head_height = self.rigid_body_pos[:, self.head_indices, 2].clone()
    feet_height = (
      self.rigid_body_pos[:, self.feet_indices, 2].clone().mean(-1).unsqueeze(-1)
    )
    head_height -= feet_height
    reward = tolerance(
      head_height,
      (self.cfg.rewards.target_head_height, np.inf),
      self.cfg.rewards.target_head_margin,
      0.1,
    )
    self.max_headheight = torch.max(
      torch.cat((head_height, self.old_headheight), dim=1), dim=1
    )[0].unsqueeze(-1)
    self.old_headheight = head_height
    return reward

  # ------------------------------------------------------------------ regularization
  def _reward_dof_acc(self):
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

  def _reward_action_rate(self):
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

  def _reward_smoothness(self):
    return torch.sum(
      torch.square(
        self.actions - self.last_actions - self.last_actions + self.last_last_actions
      ),
      dim=1,
    )

  def _reward_torques(self):
    return torch.sum(torch.square(self.torques), dim=1)

  def _reward_joint_power(self):
    return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

  def _reward_dof_vel(self):
    return torch.sum(torch.square(self.dof_vel), dim=1)

  def _reward_joint_tracking_error(self):
    return torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=-1)

  def _reward_dof_pos_limits(self):
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

  def _reward_dof_vel_limits(self):
    return torch.sum(
      (
        torch.abs(self.dof_vel)
        - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
      ).clip(min=0.0, max=1.0),
      dim=1,
    )

  # ------------------------------------------------------------------ style
  def _reward_hip_yaw_deviation(self):
    a = torch.abs(self.dof_pos[:, self.hip_joint_indices])
    return (torch.max(a, dim=-1)[0] > 1.4) | (torch.min(a, dim=-1)[0] > 0.9)

  def _reward_hip_roll_deviation(self):
    a = torch.abs(self.dof_pos[:, self.hip_roll_joint_indices])
    return (torch.max(a, dim=-1)[0] > 1.4) | (torch.min(a, dim=-1)[0] > 0.9)

  def _reward_left_foot_displacement(self):
    base_xy = self.root_states[:, :2].clone()
    left_foot_xy = self.rigid_body_pos[:, self.left_foot_indices, :2].squeeze(1)
    mse = torch.sum(torch.square(base_xy - left_foot_xy), dim=-1)
    mse_error = mse.clamp(0.3, np.inf)
    reward = torch.exp(mse_error * self.cfg.rewards.left_foot_displacement_sigma) * (
      self.rigid_body_pos[:, self.left_foot_indices, 2] < 0.15
    ).squeeze(1)
    standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return reward * standup

  def _reward_right_foot_displacement(self):
    base_xy = self.root_states[:, :2].clone()
    right_foot_xy = self.rigid_body_pos[:, self.right_foot_indices, :2].squeeze(1)
    mse = torch.sum(torch.square(base_xy - right_foot_xy), dim=-1)
    mse_error = mse.clamp(0.3, np.inf)
    reward = torch.exp(mse_error * self.cfg.rewards.right_foot_displacement_sigma) * (
      self.rigid_body_pos[:, self.right_foot_indices, 2] < 0.15
    ).squeeze(1)
    standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return reward * standup

  def _reward_knee_deviation(self):
    k = self.dof_pos[:, self.knee_joint_indices]
    return (torch.max(torch.abs(k), dim=-1)[0] > 1.65) | (
      torch.min(k, dim=-1)[0] < -0.6
    )

  def _reward_ground_parallel(self):
    left_ankle_pos = self.rigid_body_pos[:, self.left_ankle_indices, 2].clone() * 10
    right_ankle_pos = self.rigid_body_pos[:, self.right_ankle_indices, 2].clone() * 10
    var = torch.mean(
      torch.concat(
        [left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1
      ),
      dim=-1,
    )
    reward = var < 0.05
    if self.cfg.constraints.post_task:
      standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
      reward = reward * ~standup + torch.ones_like(reward) * standup
    return reward

  def _reward_feet_distance(self):
    left_foot_pos = self.rigid_body_pos[:, self.left_foot_indices, :3].clone()
    right_foot_pos = self.rigid_body_pos[:, self.right_foot_indices, :3].clone()
    feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=-1)
    return (feet_distances > 0.45).squeeze(1)

  def _reward_style_ang_vel_xy(self):
    base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
    return (
      torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2)
      * base_height
    )

  # ------------------------------------------------------------------ post-task
  def _reward_ang_vel_xy(self):
    base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return (
      torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2)
      * base_height
    )

  def _reward_lin_vel_xy(self):
    base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return (
      torch.exp(torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * -5)
      * base_height
    )

  def _reward_feet_height_var(self):
    left_foot_height = self.rigid_body_pos[:, self.left_foot_indices, 2].clone() * 10
    right_foot_height = self.rigid_body_pos[:, self.right_foot_indices, 2].clone() * 10
    feet_distance = (
      torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
    )
    standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return torch.exp(feet_distance * -2) * standup

  def _reward_target_orientation(self):
    standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return (
      torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -5)
      * standup
    )

  def _reward_target_base_height(self):
    base_height = self.root_states[:, 2]
    standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
    return (
      torch.exp(torch.abs(base_height - self.cfg.rewards.base_height_target) * -20)
      * standup
    )

  # ------------------------------------------------------------------ monitoring
  # Additions for this port (not in the release): read-only statistics for logging.
  def _init_monitor(self):
    N, dev = self.num_envs, self.device
    self.peak_torque = torch.zeros(N, self.num_dof, device=dev)
    self.peak_joint_vel = torch.zeros(N, self.num_dof, device=dev)
    self.ep_power = torch.zeros(N, device=dev)
    self.ever_stood = torch.zeros(N, dtype=torch.bool, device=dev)
    self.fell_after_stand = torch.zeros(N, dtype=torch.bool, device=dev)
    self.stand_time = torch.full((N,), -1.0, device=dev)
    self.ep_max_head = torch.zeros(N, device=dev)
    self.applied_force_steps = torch.zeros(N, device=dev)
    self.reset_stats = {}
    self.reset_stats_step = -1
    self.dof_vel_out = torch.zeros(N, dtype=torch.bool, device=dev)
    self.base_vel_out = torch.zeros(N, dtype=torch.bool, device=dev)

  def _substep_monitor(self):
    self.peak_torque_step = torch.maximum(self.peak_torque_step, self.torques.abs())
    self.step_energy += (self.dof_vel.abs() * self.torques.abs()).sum(
      -1
    ) * self.cfg.sim.dt
    self.peak_torque = torch.maximum(self.peak_torque, self.torques.abs())
    self.peak_joint_vel = torch.maximum(self.peak_joint_vel, self.dof_vel.abs())
    self.ep_power += (self.dof_vel.abs() * self.torques.abs()).sum(-1) * self.cfg.sim.dt
    self.applied_force_steps += (
      self.pending_force[:, 0, 2] > 0
    ).float() / self.cfg.control.decimation

  def update_success_monitor(self):
    """Mini-Pi version of the HoST eval criterion (see host/evaluate.py)."""
    h = self.root_states[:, 2]
    up = self.projected_gravity[:, 2] < -0.8
    stand = (h > STAND_HEIGHT) & up
    newly = stand & ~self.ever_stood
    t = (self.real_episode_length_buf.float() - self.unactuated_time) * self.dt
    self.stand_time = torch.where(newly, t, self.stand_time)
    self.ever_stood |= stand
    self.fell_after_stand |= self.ever_stood & (h < FALL_HEIGHT)
    self.ep_max_head = torch.maximum(self.ep_max_head, self.old_headheight[:, 0])

  def _log_reset_metrics(self, env_ids):
    tout = self.time_out_buf[env_ids]
    s = {
      "peak_torque": self.peak_torque[env_ids].max(-1).values.mean(),
      "peak_torque_max": self.peak_torque[env_ids].max(),
      "peak_joint_vel": self.peak_joint_vel[env_ids].max(-1).values.mean(),
      "peak_joint_vel_max": self.peak_joint_vel[env_ids].max(),
      "ep_energy": self.ep_power[env_ids].mean(),
      "success": (self.ever_stood[env_ids] & ~self.fell_after_stand[env_ids])
      .float()
      .mean(),
      "ever_stood": self.ever_stood[env_ids].float().mean(),
      "max_head_height": self.ep_max_head[env_ids].mean(),
      "final_head_height": self.old_headheight[env_ids].mean(),
      "final_base_height": self.root_states[env_ids, 2].mean(),
      "force_on_frac": (
        self.applied_force_steps[env_ids]
        / (self.episode_length_buf[env_ids].float() + 1e-6)
      ).mean(),
      "term_timeout": tout.float().mean(),
      "term_dof_vel": self.dof_vel_out[env_ids].float().mean(),
      "term_base_vel": self.base_vel_out[env_ids].float().mean(),
    }
    st = self.stand_time[env_ids]
    s["time_to_stand"] = (
      st[st >= 0].mean() if (st >= 0).any() else torch.tensor(float("nan"))
    )
    s["per_joint_peak_torque"] = self.peak_torque[env_ids].max(0).values
    self.reset_stats = s
    self.reset_stats_step = self.common_step_counter
    for buf in (self.peak_torque, self.peak_joint_vel):
      buf[env_ids] = 0.0
    self.ep_power[env_ids] = 0.0
    self.ever_stood[env_ids] = False
    self.fell_after_stand[env_ids] = False
    self.stand_time[env_ids] = -1.0
    self.ep_max_head[env_ids] = 0.0
    self.applied_force_steps[env_ids] = 0.0


# Mini-Pi counterpart of HoST eval_ground.py's G1 thresholds (stand: base > 0.7 m,
# fall: base < 0.5 m, with G1 base_height_target 0.75 m), scaled by Mini-Pi's
# base_height_target 0.34 m. An upright check (projected gravity z < -0.8, the pull
# force's own gate) is added so a high but tilted base does not count.
STAND_HEIGHT = 0.7 / 0.75 * PiCfg.rewards.base_height_target  # 0.317 m
FALL_HEIGHT = 0.5 / 0.75 * PiCfg.rewards.base_height_target  # 0.227 m
