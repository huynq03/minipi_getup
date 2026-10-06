"""FTSR Mini-Pi environment configurations.

- ``ftsr_env_cfg()``: the full task (``Mjlab-FTSR-MiniPi``). Fallen resets (supine,
  prone, left, right), Eq. 4 assistance until iteration 3000, and height-progressive
  stage-wise rewards switched by population statistics.
- ``ftsr_walk_env_cfg()``: the r_w pretraining task (``Mjlab-FTSR-MiniPi-Walk``).
  Standing resets, walking stage fixed, no assistance, fall termination. It has the
  same observation and action layout, so its checkpoint initializes the full task.

Observation groups (dimensions for Mini-Pi's 12 joints):

- ``actor``: o_t (47) = ang vel 3, projected gravity 3, command 3, gait phase 2,
  joint pos 12, joint vel 12, last action 12. Paper: o_t in R^34 for 8 DOF, no
  gait phase (a wheeled robot needs no gait clock).
- ``student``: o_{t:t-4}, the last H = 5 o_t (235), term-major.
- ``teacher``: x_t (36) = foot contact forces 6, torso height 1, root quat 4, root
  lin vel 3, root ang vel 3, body heights 13, assist wrench 6.
- ``critic``: s_t (51) = noise-free o_t 47, base lin vel 3, torso height 1. Flat
  ground, so no height map.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.rewards import feet_air_time, feet_slip
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from minipi_getup.ftsr import mdp
from minipi_getup.ftsr.config.robot import (
  MINIPI_WEIGHT,
  OPERATIONAL_TORQUE_LIMIT,
  STANCE_COM_HEIGHT,
  STANCE_FEET_WIDTH,
  get_ftsr_robot_cfg,
)
from minipi_getup.ftsr.config.stage_rewards import (
  ANG_VEL_KERNEL,
  HEIGHT_KERNEL,
  LIN_VEL_KERNEL,
  STAGE_HEIGHTS,
  STAGE_WEIGHTS,
  TRACKING_MIN_HEIGHT,
)
from minipi_getup.getup.mdp.force_guidance import low_for_too_long

STEPS_PER_ITERATION = 24

# Relative joint-position action: q_target = q + ACTION_SCALE * a. The peak P torque
# per unit action is kp * 0.13 (5.5 Nm at kp 42).
ACTION_SCALE = 0.13
SETTLE_STEPS = 25  # 0.5 s with actions (and assist) held after a fallen reset.
EPISODE_LENGTH_S = 20.0  # As getup_gym.
# Gait clock period (s) of the walking stage, from the tuned Mini-Pi velocity task.
GAIT_PERIOD = 0.5

# Eq. 4 assistance, scaled to Mini-Pi (audit Sec. 3):
# - F_max = 0.75 m g: never the whole weight, so even at full strength the legs must
#   carry part of the body. G1 in the released code used ~1.0 m g with a linear
#   profile.
# - mu = 15 /m, i.e. mu x stance height ~ 5: at the supine height (0.085 m) toward
#   h1 = 0.19 m the factor is 0.79, toward the stance 0.98.
# - T_max: 30% of the gravity moment m g h_com at 90 deg tilt, i.e.
#   T_max * pi/2 = 0.3 m g h_com, so T_max = 2.9 Nm/rad.
# - The assistance ends at iteration 3000 (paper).
ASSIST_F_MAX = 0.75 * MINIPI_WEIGHT
ASSIST_MU = 15.0
ASSIST_T_MAX = 0.3 * MINIPI_WEIGHT * STANCE_COM_HEIGHT / (3.14159 / 2)
ASSIST_END_ITERATION = 3000

FALLEN_POSE_NAMES = ("supine", "prone", "left_side", "right_side")

_TORSO = SceneEntityCfg("robot", body_names=("base_link",))
_FEET = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
_FEET_SITES = SceneEntityCfg("robot", site_names=("r_foot", "l_foot"))
_FEET_SENSOR = "feet_ground_contact"


def _observations() -> dict[str, ObservationGroupCfg]:
  def o_t(noisy: bool) -> dict[str, ObservationTermCfg]:
    def n(lo: float):
      return Unoise(n_min=-lo, n_max=lo) if noisy else None

    return {
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=n(0.2),
        scale=0.25,
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity, noise=n(0.05)
      ),
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
        scale=(2.0, 2.0, 0.25),
      ),
      "gait_phase": ObservationTermCfg(
        func=mdp.gait_phase,
        params={"period": GAIT_PERIOD, "command_name": "twist"},
      ),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel, noise=n(0.03), params={"biased": noisy}
      ),
      "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=n(1.5), scale=0.05),
      "actions": ObservationTermCfg(func=mdp.last_action),
    }

  teacher = {
    "feet_forces": ObservationTermCfg(
      func=mdp.feet_contact_forces,
      params={"sensor_name": _FEET_SENSOR, "scale": 1.0 / MINIPI_WEIGHT},
    ),
    "torso_height": ObservationTermCfg(
      func=mdp.torso_height, scale=2.0, params={"asset_cfg": _TORSO}
    ),
    "root_quat": ObservationTermCfg(func=mdp.root_quat_canonical),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=2.0),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25),
    "body_heights": ObservationTermCfg(func=mdp.body_heights, scale=2.0),
    "assist_wrench": ObservationTermCfg(
      func=mdp.assist_wrench,
      params={"force_scale": 1.0 / ASSIST_F_MAX, "torque_scale": 1.0 / ASSIST_T_MAX},
    ),
  }
  critic = {
    **o_t(noisy=False),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=2.0),
    "torso_height": ObservationTermCfg(
      func=mdp.torso_height, scale=2.0, params={"asset_cfg": _TORSO}
    ),
  }
  return {
    "actor": ObservationGroupCfg(terms=o_t(noisy=True), enable_corruption=True),
    "student": ObservationGroupCfg(
      terms=o_t(noisy=True), enable_corruption=True, history_length=5
    ),
    "teacher": ObservationGroupCfg(terms=teacher, enable_corruption=False),
    "critic": ObservationGroupCfg(terms=critic, enable_corruption=False),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  """All stage terms; weights start at stage r_u and the stage manager moves them."""
  terms = {
    "track_lin_vel": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=0.0,
      params={
        "coef": LIN_VEL_KERNEL,
        "command_name": "twist",
        "min_height": TRACKING_MIN_HEIGHT,
        "asset_cfg": _TORSO,
      },
    ),
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=0.0,
      params={
        "coef": ANG_VEL_KERNEL,
        "command_name": "twist",
        "min_height": TRACKING_MIN_HEIGHT,
        "asset_cfg": _TORSO,
      },
    ),
    "orientation": RewardTermCfg(func=mdp.orientation_xy_norm, weight=0.0),
    "upside_down": RewardTermCfg(func=mdp.upside_down, weight=0.0),
    "torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=0.0),
    "dof_acc": RewardTermCfg(func=mdp.joint_acc_fd_l2, weight=0.0),
    "dof_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=0.0),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=0.0),
    "action_smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight=0.0),
    "dof_pos": RewardTermCfg(func=mdp.joint_deviation_l2, weight=0.0),
    "dof_energy": RewardTermCfg(func=mdp.joint_power_sq, weight=0.0),
    "base_height": RewardTermCfg(
      func=mdp.stage_height_exp,
      weight=0.0,
      params={
        "coef": HEIGHT_KERNEL,
        "default_height_cmd": STAGE_HEIGHTS[0],
        "asset_cfg": _TORSO,
      },
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=0.0),
    "feet_distance": RewardTermCfg(
      func=mdp.feet_lateral_distance,
      weight=0.0,
      params={"target": STANCE_FEET_WIDTH, "asset_cfg": _FEET},
    ),
    "leg_bias": RewardTermCfg(func=mdp.leg_mirror_error, weight=0.0),
    "no_fly": RewardTermCfg(
      func=mdp.no_feet_contact, weight=0.0, params={"sensor_name": _FEET_SENSOR}
    ),
    "feet_support": RewardTermCfg(
      func=mdp.feet_support_force,
      weight=0.0,
      params={
        "sensor_name": _FEET_SENSOR,
        "robot_weight": MINIPI_WEIGHT,
        "width": 0.2 * MINIPI_WEIGHT,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=feet_air_time,
      weight=0.0,
      params={
        "sensor_name": _FEET_SENSOR,
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "feet_slip": RewardTermCfg(
      func=feet_slip,
      weight=0.0,
      params={
        "sensor_name": _FEET_SENSOR,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": _FEET_SITES,
      },
    ),
    "feet_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.0,
      params={
        "sensor_name": _FEET_SENSOR,
        "period": GAIT_PERIOD,
        "offset": (0.0, 0.5),
        "stance_fraction": 0.55,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "torque_limit": RewardTermCfg(
      func=mdp.torque_limit_excess,
      weight=0.0,
      params={"limit": 0.8 * OPERATIONAL_TORQUE_LIMIT},
    ),
  }
  assert set(terms) == set(STAGE_WEIGHTS)
  for name, term in terms.items():
    term.weight = STAGE_WEIGHTS[name][0]
  return terms


def _domain_randomization() -> dict[str, EventTermCfg]:
  """Paper Table III adapted to Mini-Pi (audit Sec. 6)."""
  return {
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    ),
    "torso_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={"asset_cfg": _TORSO, "operation": "add", "ranges": (-0.1, 0.5)},
    ),
    "torso_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": _TORSO,
        "operation": "add",
        "ranges": {0: (-0.015, 0.015), 1: (-0.01, 0.01), 2: (-0.015, 0.015)},
      },
    ),
    "friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.3, 1.5),
        "shared_random": True,
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={"kp_range": (0.95, 1.05), "kd_range": (0.95, 1.05)},
    ),
    "motor_strength": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={"effort_limit_range": (0.85, 1.05)},
    ),
  }


def _base_cfg(play: bool) -> ManagerBasedRlEnvCfg:
  feet_contact = ContactSensorCfg(
    name=_FEET_SENSOR,
    primary=ContactMatch(
      mode="subtree", pattern=r"^(r_ankle_roll_link|l_ankle_roll_link)$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  twist = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.1,
    rel_heading_envs=0.0,
    heading_command=False,
    # Paper (wheeled): x [-0.5, 0.8], y 0, yaw [-0.3, 0.3] at a 0.75 m stance. Mini-Pi:
    # x scaled by the 0.46 length ratio; yaw a little wider for the smaller robot.
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.3, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5)
    ),
  )
  twist.viz.z_offset = 0.3

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
      entities={"robot": get_ftsr_robot_cfg()},
      sensors=(feet_contact,),
    ),
    observations=_observations(),
    actions={
      "joint_pos": mdp.FtsrJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=ACTION_SCALE,
        settle_steps=SETTLE_STEPS,
        operational_torque_limit=OPERATIONAL_TORQUE_LIMIT,
      )
    },
    commands={"twist": twist},
    events={},
    rewards=_rewards(),
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "nan": TerminationTermCfg(func=mdp.nan_detection),
    },
    curriculum={},
    metrics={
      "standing": MetricsTermCfg(
        func=mdp.is_standing, reduce="mean", params={"asset_cfg": _TORSO}
      ),
      "stood_up": MetricsTermCfg(
        func=mdp.is_standing, reduce="max", params={"asset_cfg": _TORSO}
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.2,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=200,
      mujoco=MujocoCfg(
        timestep=0.002, iterations=10, ls_iterations=20, impratio=10, cone="elliptic"
      ),
    ),
    decimation=10,
    episode_length_s=EPISODE_LENGTH_S,
  )
  cfg.events = _domain_randomization()
  if play:
    for group in ("actor", "student"):
      cfg.observations[group].enable_corruption = False
  return cfg


def _stage_event(fixed_stage: int | None) -> EventTermCfg:
  return EventTermCfg(
    func=mdp.ftsr_stage_manager,
    mode="step",
    params={
      "heights": STAGE_HEIGHTS,
      "weights": STAGE_WEIGHTS,
      "fraction": 2.0 / 3.0,
      "steps_per_iteration": STEPS_PER_ITERATION,
      "fixed_stage": fixed_stage,
      "asset_cfg": _TORSO,
    },
  )


def ftsr_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play)
  n = len(FALLEN_POSE_NAMES)
  cfg.events["reset_pose"] = EventTermCfg(
    func=mdp.reset_pose_distribution,
    mode="reset",
    params={
      "poses": tuple(mdp.FALLEN_POSES[name] for name in FALLEN_POSE_NAMES),
      "probabilities": (1.0 / n,) * n,
      "euler_noise": 0.3,
      "joint_noise": 0.3,
    },
  )
  # Stage first, so the assist reads this step's target height.
  cfg.events["stage"] = _stage_event(fixed_stage=None)
  cfg.events["assist"] = EventTermCfg(
    func=mdp.ftsr_assist,
    mode="step",
    params={
      "f_max": ASSIST_F_MAX,
      "t_max": ASSIST_T_MAX,
      "mu": ASSIST_MU,
      "end_iteration": ASSIST_END_ITERATION,
      "robot_weight": MINIPI_WEIGHT,
      "steps_per_iteration": STEPS_PER_ITERATION,
      "default_height_cmd": STAGE_HEIGHTS[0],
      "settle_steps": SETTLE_STEPS,
      "asset_cfg": _TORSO,
    },
  )
  # Paper: an agent still on the ground after 10 s is terminated (and penalized by
  # the stage's termination weight).
  cfg.terminations["low_too_long"] = TerminationTermCfg(
    func=low_for_too_long,
    params={"max_time_s": 10.0, "min_height": 0.15, "asset_cfg": _TORSO},
  )
  if play:
    # The assistance exists only in simulation training.
    del cfg.events["assist"]
  return cfg


def ftsr_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """r_w pretraining (paper Sec. III-A, "Model Initialization")."""
  cfg = _base_cfg(play)
  cfg.events["reset_pose"] = EventTermCfg(
    func=mdp.reset_pose_distribution,
    mode="reset",
    params={
      "poses": (mdp.STANDING_POSE,),
      "probabilities": (1.0,),
      "euler_noise": 0.05,
      "joint_noise": 0.05,
    },
  )
  cfg.events["stage"] = _stage_event(fixed_stage=2)
  for name, term in cfg.rewards.items():
    term.weight = STAGE_WEIGHTS[name][2]
  # Walking only: a fall ends the episode (the recovery is learned later).
  cfg.terminations["fell"] = TerminationTermCfg(
    func=mdp.fell_over,
    params={"min_height": 0.2, "max_tilt": 1.0, "asset_cfg": _TORSO},
  )
  return cfg
