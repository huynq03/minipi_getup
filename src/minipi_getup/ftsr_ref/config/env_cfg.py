"""Environment configurations of the two reference tasks.

- ``recovery_env_cfg()`` (``Mjlab-FTSR-Ref-MiniPi-Recovery``): four fallen poses,
  passive window, Eq. 4 assistance until iteration 3000, stage-wise rewards r_u -> r_s
  -> r_w from population heights. No termination except the 20 s time-out (release).
- ``walk_env_cfg()`` (``Mjlab-FTSR-Ref-MiniPi-Walk``): the paper's model
  initialization (Sec. III-A, absent from the release: PAPER_COMPLETION). Standing
  resets, stage fixed at r_w, no assistance, falls terminate.

Both use the same plant, action path, observation layout and networks, so the
walking checkpoint initializes the recovery run unchanged.

Derived numbers (FTSR_REFERENCE_AUDIT_V2.md Sec. 10; L = 0.345 m, L_ref = 0.75 m):

- Stage target heights = thresholds (paper: S_1 = {h > h1_cmd}, S_2 = {h > h2_cmd}):
  h1 = 0.19 m: the release's r_s switch at 0.40 m (0.533 L_ref -> 0.184 m) and
  Mini-Pi's lowest statically balanced crouch (0.19 m) agree; the crouch is used.
  h2 = 0.276 m: 0.8 L, the release's "upright" level (target ladder 0.6 m and the
  0.6 m velocity-tracking gate). h3 = 0.345 m: the stance (release 0.75 m).
- F_max = 400 N / (27.669 kg g) x m g = 1.474 m g = 100.3 N (dimensionless reference
  scaling; ablation at 1.0 m g).
- mu (UNRESOLVED, calibration candidate): match the release's height factor at its
  reset state, (0.75 - 0.1) / 0.75 = 0.867, at Mini-Pi lying (0.085 m) under r_u
  (h1 = 0.19): mu = -ln(1 - 0.867) / 0.105 = 19.2 /m.
- T_max (UNRESOLVED, calibration candidate): match the release torque magnitude at its
  reset state, 10 |q_y + 1| x 0.867 = 2.54 N m = 0.0125 m_ref g L_ref; for Mini-Pi
  0.0125 m g L = 0.293 N m at tilt pi/2 and factor 0.867: T_max = 0.215 N m/rad.
- Passive window: release 30 policy steps = 0.6 s ("action 0 = hold default pose").
  Mini-Pi in the deploy Passive state (kp 0, kd 1) needs ~2 s after the reset drop to
  come to rest (validate.py test 22: at 0.6 s the side poses still roll at p95
  0.3 m/s; at 2.0 s every pose is below 0.03 m/s and 0.4 rad/s). The paper requires
  settled initial states, so the window is 100 steps = 2.0 s (ROBOT_ADAPTATION).
- Commands: release vx [-0.5, 0.8], vy 0, wz [-0.3, 0.3], resampled every 10 s, no
  standing envs; Froude-scaled (sqrt(L/L_ref) = 0.678): vx [-0.34, 0.54],
  wz [-0.44, 0.44].
- Action scale per joint: the release's functional analog where one exists (hip roll
  0.2, femur/hip pitch 0.6, tibia/knee 0.6); otherwise the release's mean
  scale-to-range ratio 0.224 times the Mini-Pi range (hip yaw 0.2, ankle pitch 0.4,
  ankle roll 0.25). raw_clip = the release's clip_actions (50).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from minipi_getup.ftsr_ref import mdp
from minipi_getup.ftsr_ref.config.rewards import FEET_SENSOR, REWARD_TABLE
from minipi_getup.ftsr_ref.config.robot import (
  LENGTH_RATIO,
  MINIPI_MASS,
  REF_MASS,
  STANCE_HEIGHT,
  get_robot_cfg,
)

STEPS_PER_ITERATION = 24
# Physics 0.5 ms x 40 = 20 ms policy step (kept from v2; mini_pi_fsm's own simulator
# also runs 0.5 ms). The vendor model has no armature (ankle-roll joint inertia
# 4.4e-4 kg m^2); the small step keeps the light joints well resolved under the plain
# 16 Nm PD (experiment pd16_noslew: no torque-speed envelope, no target slew).
PHYSICS_DT = 0.0005
DECIMATION = 40
EPISODE_LENGTH_S = 20.0
PASSIVE_STEPS = 100  # 2.0 s settle in deploy Passive (release: 30 steps = 0.6 s)
STAGE_HEIGHTS = (0.19, 0.6 * LENGTH_RATIO, STANCE_HEIGHT)
# Height-reward targets (recovery v2). Only r_u differs: its target lies above the S_1
# threshold h1 by the release's ratio of the height target active at its g2->g3 switch
# to that switch (0.45 m / 0.40 m), 0.19 * 1.125 = 0.214 m. The v1 run (target = h1)
# settled just below h1 (experiment log). Eq. 4 and the thresholds keep STAGE_HEIGHTS.
STAGE_REWARD_HEIGHTS = (STAGE_HEIGHTS[0] * 0.45 / 0.40, *STAGE_HEIGHTS[1:])

ACTION_SCALE = {
  r".*_hip_pitch_joint": 0.6,
  r".*_hip_roll_joint": 0.2,
  r".*_thigh_joint": 0.2,
  r".*_calf_joint": 0.6,
  r".*_ankle_pitch_joint": 0.4,
  r".*_ankle_roll_joint": 0.25,
}
RAW_CLIP = 50.0
# Soft joint-speed envelope (reward ``qd_soft_envelope`` and monitoring), rad/s.
# pd16_noslew: 6.28 (was 3.0 with the slew contract). Not a clamp.
QD_SOFT_LIMIT = 6.28

ASSIST_F_MAX = 400.0 / REF_MASS * MINIPI_MASS  # 100.3 N = 1.474 m g
ASSIST_MU = -math.log(1.0 - 0.65 / 0.75) / (STAGE_HEIGHTS[0] - 0.085)
ASSIST_T_MAX = (
  (10.0 * (1.0 - math.sqrt(0.5)) * 0.65 / 0.75 / (REF_MASS * 9.81 * 0.75))
  * (MINIPI_MASS * 9.81 * STANCE_HEIGHT)
  / ((0.65 / 0.75) * math.pi / 2.0)
)
ASSIST_END_ITERATION = 3000

COMMAND_RANGES = {
  "lin_vel_x": (-0.5 * math.sqrt(LENGTH_RATIO), 0.8 * math.sqrt(LENGTH_RATIO)),
  "lin_vel_y": (0.0, 0.0),
  "ang_vel_z": (-0.3 / math.sqrt(LENGTH_RATIO), 0.3 / math.sqrt(LENGTH_RATIO)),
}


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
  noisy = not play
  return {
    # o_t; must come first (the history term reads the frame it caches).
    "actor": ObservationGroupCfg(
      terms={
        "o_t": ObservationTermCfg(func=mdp.ftsr_obs_frame, params={"noisy": noisy})
      },
      enable_corruption=False,
    ),
    "policy": ObservationGroupCfg(
      terms={"history": ObservationTermCfg(func=mdp.ftsr_history)},
      enable_corruption=False,
    ),
    "teacher": ObservationGroupCfg(
      terms={
        "x_t": ObservationTermCfg(
          func=mdp.ftsr_teacher_obs, params={"sensor_name": FEET_SENSOR}
        )
      },
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={"s_t": ObservationTermCfg(func=mdp.ftsr_critic_obs)},
      enable_corruption=False,
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  terms = {}
  for name, (weights, params) in REWARD_TABLE.items():
    terms[name] = RewardTermCfg(
      func=getattr(mdp, name),
      weight=1.0,
      params={"stage_weights": weights, **params},
    )
  return terms


def _base_cfg(play: bool) -> ManagerBasedRlEnvCfg:
  feet_contact = ContactSensorCfg(
    name=FEET_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(r_ankle_roll_link|l_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  twist = mdp.FtsrRefVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    heading_command=False,
    ranges=UniformVelocityCommandCfg.Ranges(**COMMAND_RANGES),
  )
  twist.viz.z_offset = 0.3
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
      entities={"robot": get_robot_cfg()},
      sensors=(feet_contact,),
    ),
    observations=_observations(play),
    actions={
      "joint_pos": mdp.FtsrActionCfg(
        scale=dict(ACTION_SCALE),
        raw_clip=RAW_CLIP,
        qd_soft=QD_SOFT_LIMIT,
      )
    },
    commands={"twist": twist},
    events={
      "plant_setup": EventTermCfg(func=mdp.plant_setup, mode="startup"),
    },
    rewards=_rewards(),
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "nan": TerminationTermCfg(func=mdp.nan_detection),
    },
    curriculum={},
    metrics={},
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
        timestep=PHYSICS_DT,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
  )


def recovery_env_cfg(
  play: bool = False, monotonic: bool = True, target_rate_limit: float = 0.0
) -> ManagerBasedRlEnvCfg:
  """``monotonic=False`` = release stateless stages (recovery v1/v2 semantics).
  ``target_rate_limit`` > 0: PD-target rate limiter (rad per policy step)."""
  cfg = _base_cfg(play)
  act = cfg.actions["joint_pos"]
  act.passive_steps = PASSIVE_STEPS
  act.target_rate_limit = target_rate_limit
  if not play:
    act.assist = mdp.AssistCfg(
      f_max=ASSIST_F_MAX,
      t_max=ASSIST_T_MAX,
      mu=ASSIST_MU,
      end_iteration=ASSIST_END_ITERATION,
      steps_per_iteration=STEPS_PER_ITERATION,
    )
  cfg.events["stage_setup"] = EventTermCfg(
    func=mdp.stage_setup,
    mode="startup",
    params={
      "stage": mdp.StageCfg(
        heights=STAGE_HEIGHTS,
        reward_heights=STAGE_REWARD_HEIGHTS,
        # Recovery v3: latched (monotonic) stages, see mdp/stages.py.
        monotonic=monotonic,
      )
    },
  )
  cfg.events["reset_pose"] = EventTermCfg(
    func=mdp.reset_pose,
    mode="reset",
    params={
      "poses": tuple(mdp.FALLEN_POSES[n] for n in mdp.FALLEN_POSE_NAMES),
      "euler_noise": 0.3,
      "joint_noise": 0.3,
    },
  )
  return cfg


def walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _base_cfg(play)
  cfg.events["stage_setup"] = EventTermCfg(
    func=mdp.stage_setup,
    mode="startup",
    params={"stage": mdp.StageCfg(heights=STAGE_HEIGHTS, fixed_stage=2)},
  )
  cfg.events["reset_pose"] = EventTermCfg(
    func=mdp.reset_pose,
    mode="reset",
    params={"poses": (mdp.STANDING_POSE,), "euler_noise": 0.05, "joint_noise": 0.05},
  )
  cfg.terminations["fell"] = TerminationTermCfg(
    func=mdp.fell_over, params={"min_height": 0.15, "max_tilt": 1.0}
  )
  return cfg
