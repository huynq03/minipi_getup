"""Mini-Pi getup with FTSR-style force guidance and height-progressive stages.

Built on ``minipi_getup_env_cfg``. Paper parts used: the height/time-scaled assist
wrench (Eq. 4), its force and torque as fixed-factor costs (Eq. 5-8), height-progressive
stage-wise rewards, the "leg bias" and "lying too long" terms, and privileged critic
inputs (assist wrench, torso height). Not used: the concurrent teacher-student encoder
(the critic is asymmetric instead), walking-reward pretraining, terrains.
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from minipi_getup.getup import mdp
from minipi_getup.getup.config.minipi.env_cfgs import (
  _SQUAT_RESET_PROBABILITY,
  _TORSO_HEIGHT,
  SQUAT_RESETS,
  SUPINE_JOINT_POS,
  SUPINE_ROOT_POS,
  SUPINE_ROOT_QUAT,
  minipi_getup_env_cfg,
)

# Total mass from cl_pai.xml (6.92 kg).
_WEIGHT = 6.92 * 9.81
# Lowest statically balanced crouch has the base at ~0.19 m (see env_cfgs.py). The
# first stage targets it; a sit on the ground is at ~0.06 m.
_SQUAT_HEIGHT = 0.20
# The assist fades to zero at this env step (iterations * 24).
ASSIST_END_STEP = 1500 * 24

_TORSO = SceneEntityCfg("robot", body_names=("base_link",))


def minipi_getup_force_guided_env_cfg(play: bool = False):
  cfg = minipi_getup_env_cfg(play=play)

  # Supine starts only: the crouch starts were a workaround for the same dead point
  # the force guidance targets.
  cfg.events["reset_supine_or_squat"].params["poses"] = (
    {
      "root_pos": SUPINE_ROOT_POS,
      "root_quat": SUPINE_ROOT_QUAT,
      "joint_pos": SUPINE_JOINT_POS,
    },
  )
  cfg.events["reset_supine_or_squat"].params["probabilities"] = (1.0,)

  # Paper Eq. 4. At the supine height (0.085 m) k = 0.68 toward the first-stage target
  # and 0.93 toward standing. 0.6 x weight on the torso lifts it, never the whole robot.
  if not play:
    cfg.events["force_guided_assist"] = EventTermCfg(
      func=mdp.force_guided_assist,
      mode="step",
      params={
        "f_max": 0.6 * _WEIGHT,
        "t_max": 3.0,
        "mu": 10.0,
        "height_cmd": _SQUAT_HEIGHT,
        "end_step": ASSIST_END_STEP,
        "asset_cfg": _TORSO,
      },
    )

  # Costs C1 = |F|, C2 = |T| with fixed penalty factors. At the supine start they cost
  # about 0.55 and 0.6 per second, against up to 2 per second of height reward.
  cfg.rewards["assist_force"] = RewardTermCfg(func=mdp.assist_force_cost, weight=-0.02)
  cfg.rewards["assist_torque"] = RewardTermCfg(func=mdp.assist_torque_cost, weight=-0.2)
  # The assist lifts the torso fast while it is low; don't charge the policy for that.
  del cfg.rewards["upward_launch"]

  cfg.rewards["leg_mirror"] = RewardTermCfg(
    func=mdp.leg_mirror_error,
    weight=-0.5,
    params={
      "same_sign": ("hip_pitch_joint", "calf_joint", "ankle_pitch_joint"),
      "mirrored": ("hip_roll_joint", "thigh_joint", "ankle_roll_joint"),
    },
  )

  # Paper: an agent still on the ground after 10 s is terminated with a penalty.
  cfg.terminations["low_too_long"] = TerminationTermCfg(
    func=mdp.low_for_too_long,
    params={"max_time_s": 10.0, "min_height": 0.15, "asset_cfg": _TORSO},
  )
  cfg.rewards["terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=-50.0)

  # Height-progressive stages: (1) get the torso up into a crouch, (2) stand. The
  # paper's third (walking) stage has no counterpart here.
  cfg.curriculum["height_stage"] = CurriculumTermCfg(
    func=mdp.height_stage,
    params={
      "height_reward_name": "torso_height",
      "asset_cfg": _TORSO,
      "stages": [
        {
          "height_cmd": _SQUAT_HEIGHT,
          "weights": {"torso_height": 2.0, "posture": 0.0, "leg_mirror": -0.5},
        },
        {
          "height_cmd": _TORSO_HEIGHT,
          "weights": {"torso_height": 3.0, "posture": 1.0, "leg_mirror": -0.05},
        },
      ],
    },
  )

  # Privileged critic inputs.
  critic = cfg.observations["critic"].terms
  critic["assist_wrench"] = ObservationTermCfg(func=mdp.assist_wrench_obs)
  critic["torso_height"] = ObservationTermCfg(
    func=mdp.body_height, params={"asset_cfg": _TORSO}
  )

  return cfg


# Base-height levels of the released humanoid config ([0.3, 0.45, 0.6, 0.75] of a 0.75 m
# stance), scaled to Mini-Pi's 0.344 m.
_RELEASED_LEVELS = tuple(round(f * _TORSO_HEIGHT, 3) for f in (0.4, 0.6, 0.8, 1.0))
# Lowest balanced crouch (env_cfgs._SQUAT_POSES[0]): the released code's knee-down pose.
_CROUCH_POSE = {
  r".*_hip_pitch_joint": -1.125,
  r".*_calf_joint": 1.625,
  r".*_ankle_pitch_joint": -0.5,
  r".*_hip_roll_joint": 0.0,
  r".*_thigh_joint": 0.0,
  r".*_ankle_roll_joint": 0.0,
}
_NOMINAL_POSE = {r".*": 0.0}


def minipi_getup_force_guided_released_env_cfg(play: bool = False):
  """Variant following the released getup_gym code rather than the paper text.

  - Assist force linear in height up to the standing height (not the stage target)
    with f_max = robot weight (350 N on the ~35 kg G1), on until iteration 1500.
  - No force/torque cost: the released code's constraint term is never fed.
  - Four target-height levels, advanced when the batch mean reaches 0.8 x the level.
  - A height-banded pose reward: deep crouch while low, nominal pose higher up.
  - No "lying too long" termination (off in the released configs).
  """
  cfg = minipi_getup_force_guided_env_cfg(play=play)

  if not play:
    cfg.events["force_guided_assist"].params.update(
      {
        "f_max": _WEIGHT,
        "height_cmd": _TORSO_HEIGHT,
        "profile": "linear",
        "follow_stage": False,
        "end_step": 1500 * 24,
      }
    )

  for name in ("assist_force", "assist_torque", "terminated"):
    del cfg.rewards[name]
  del cfg.terminations["low_too_long"]

  # Released bands for G1: knee-down in 0.1-0.4 m, nominal in 0.4-0.7 m of a 0.75 m
  # stance. Here the low band covers lying (0.085 m), sitting (0.064 m) and the lowest
  # balanced crouch (0.19 m); scaling G1's 0.4 m would cut at 0.18 m, below it.
  cfg.rewards["banded_pose"] = RewardTermCfg(
    func=mdp.height_banded_pose,
    weight=2.0,
    params={
      "bands": [
        (0.04, 0.21, _CROUCH_POSE),
        (0.21, 0.93 * _TORSO_HEIGHT, _NOMINAL_POSE),
      ],
      "std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "height_cfg": _TORSO,
    },
  )

  low, mid, high, stand = _RELEASED_LEVELS
  cfg.curriculum["height_stage"].params.update(
    {
      "rule": "mean",
      "mean_factor": 0.8,
      "stages": [
        {"height_cmd": low, "weights": {"torso_height": 2.0, "leg_mirror": -0.5}},
        {"height_cmd": mid, "weights": {"torso_height": 2.0, "leg_mirror": -0.5}},
        {"height_cmd": high, "weights": {"torso_height": 3.0, "leg_mirror": -0.05}},
        {"height_cmd": stand, "weights": {"torso_height": 3.0, "leg_mirror": -0.05}},
      ],
    }
  )
  return cfg


STAND_RESUME_ITER = 3000
STAND_ASSIST_ITERS = 750


def minipi_getup_force_guided_stand_env_cfg(play: bool = False):
  """Fine-tune stage after the released variant: crouch-to-stand shaping plus assist.

  The released variant's policy reaches and holds the lowest crouch (0.19 m) without
  assist but stops there: up to 0.21 m the banded pose reward still pays for the crouch
  pose. Here the crouch band ends at 0.15 m, so from the crouch up only the nominal
  (standing) pose pays. Without assist the crouch stayed an attractor (400 iterations,
  no stand), although a scripted joint interpolation stands up from it with ~3 Nm, so
  the assist comes back for this second dead point. Resume from the released run's
  ``model_2999``: a resumed run restores the step counter (3000 * 24), so the assist
  starts there at full strength and fades out over ``STAND_ASSIST_ITERS``.
  """
  cfg = minipi_getup_force_guided_released_env_cfg(play=play)
  if not play:
    cfg.events["force_guided_assist"].params.update(
      {
        "start_step": STAND_RESUME_ITER * 24,
        "end_step": (STAND_RESUME_ITER + STAND_ASSIST_ITERS) * 24,
      }
    )
  # The released variant's nominal band ends at 0.32 m, below the 0.344 m stance, so
  # standing itself earned no pose reward; extend it past the stance.
  cfg.rewards["banded_pose"].params["bands"] = [
    (0.04, 0.15, _CROUCH_POSE),
    (0.15, 0.5, _NOMINAL_POSE),
  ]
  # The released policy parks hip pitch and knee on their hard stops (-1.25, 1.65) by
  # pushing the clamped relative target past them: a stance that needs no balancing
  # and that exploration noise cannot leave. Charge for the push and for sitting
  # within 0.1 rad of a limit (hip roll excluded: hip_roll_open asks for its limit).
  cfg.rewards["target_past_limits"] = RewardTermCfg(
    func=mdp.target_past_limits, weight=-2.0
  )
  cfg.rewards["joint_near_limits"] = RewardTermCfg(
    func=mdp.joint_near_limits,
    weight=-5.0,
    params={
      "margin": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r".*_hip_pitch_joint",
          r".*_thigh_joint",
          r".*_calf_joint",
          r".*_ankle_pitch_joint",
          r".*_ankle_roll_joint",
        ),
      ),
    },
  )
  # Also start 40% of episodes in the balanced crouches (as the base task does), so the
  # crouch-to-stand step gets practice without first replaying the whole get-up.
  supine = cfg.events["reset_supine_or_squat"].params["poses"][0]
  squat_prob = _SQUAT_RESET_PROBABILITY / len(SQUAT_RESETS)
  cfg.events["reset_supine_or_squat"].params["poses"] = (supine, *SQUAT_RESETS)
  cfg.events["reset_supine_or_squat"].params["probabilities"] = (
    1.0 - _SQUAT_RESET_PROBABILITY,
    *(squat_prob,) * len(SQUAT_RESETS),
  )
  return cfg


# The stand fine-tune (ftsr_stand_v4) gets up from supine in ~1 s by leaping: joint
# speeds to 22 rad/s, torque at the 16 Nm cap, base vertical speed 2.3 m/s and
# roll/pitch rate 8.7 rad/s. Hinge penalties on each, above hardware-friendly limits,
# ramped in from SAFE_RESUME_ITER so the get-up slows down instead of being abandoned.
SAFE_RESUME_ITER = 3850
_SAFE_LIMITS = {
  # name: (func, limit, full weight)
  # safe_v5-v7 used 4 rad/s, 10 Nm, 0.3 m/s, 2 rad/s with actions clipped to +-0.6;
  # under those the policy could not sit up from supine at all (the old get-up swings
  # the legs at ~12 rad/s to sit up), so v8 loosens them.
  "joint_vel_excess": (mdp.joint_vel_excess_l2, 8.0, -0.05),
  "torque_excess": (mdp.actuator_force_excess_l2, 12.0, -0.05),
  "base_vz_excess": (mdp.base_lin_vel_z_excess_l2, 0.5, -20.0),
  "base_ang_vel_excess": (mdp.base_ang_vel_xy_excess_l2, 3.0, -2.0),
}


# With clipped actions the policy could no longer leave the supine pose (safe_v6_clip:
# 0% from supine), so the assist comes back, fading out over SAFE_ASSIST_ITERS from
# SAFE_ASSIST_START (v7: 4950 + 750; v8 resumes safe_v5's model_4350).
SAFE_ASSIST_START = 4350
SAFE_ASSIST_ITERS = 500


def minipi_getup_force_guided_safe_env_cfg(play: bool = False):
  """Stand fine-tune plus speed/torque limits, for a slower, hardware-safe get-up."""
  cfg = minipi_getup_force_guided_stand_env_cfg(play=play)
  if not play:
    cfg.events["force_guided_assist"].params.update(
      {
        "start_step": SAFE_ASSIST_START * 24,
        "end_step": (SAFE_ASSIST_START + SAFE_ASSIST_ITERS) * 24,
      }
    )
  for name, (func, limit, weight) in _SAFE_LIMITS.items():
    cfg.rewards[name] = RewardTermCfg(
      func=func, weight=weight / 3.0, params={"limit": limit}
    )
    cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": name,
        "stages": [
          {"step": 0, "weight": weight / 3.0},
          {"step": (SAFE_RESUME_ITER + 150) * 24, "weight": 2.0 * weight / 3.0},
          {"step": (SAFE_RESUME_ITER + 300) * 24, "weight": weight},
        ],
      },
    )
  return cfg
