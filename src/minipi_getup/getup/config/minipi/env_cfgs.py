"""HighTorque Mini-Pi getup environment configuration.

First stage: recover from one fixed supine (lying on the back) pose to standing.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from minipi_getup.asset_zoo.robots.hightorque_minipi.minipi_constants import (
  get_minipi_robot_cfg,
)
from minipi_getup.getup import mdp
from minipi_getup.getup.getup_env_cfg import make_getup_env_cfg
from minipi_getup.getup.mdp.actions import SettleRelativeJointPositionActionCfg

# Measured by settling the home keyframe (all joints 0) on flat ground. base_link is
# the single torso body; its origin sits on the hip line. There is no separate pelvis
# height reward: the hip_pitch_link origins are rigidly 33 mm below base_link, so it
# would duplicate torso_height.
_TORSO_HEIGHT = 0.344
# Forward offset of the ankle_roll_link origins from base_link in the standing pose,
# measured from the home keyframe.
_FOOT_FORWARD_OFFSET = 0.055

# Fixed supine start: legs in the nominal stance shape, lying on the back. Measured at
# rest: the robot lies on the rear torso capsule and both heels, pitched -88.3 deg
# about y (front facing up) with the base origin 84 mm above the ground. Re-measure
# the root height/pitch if the joint pose is changed.
SUPINE_JOINT_POS: dict[str, float] = {
  r".*_hip_pitch_joint": 0.0,
  r".*_hip_roll_joint": 0.0,
  r".*_thigh_joint": 0.0,
  r".*_calf_joint": 0.0,
  r".*_ankle_pitch_joint": 0.0,
  r".*_ankle_roll_joint": 0.0,
}
_SUPINE_PITCH = -1.5418
SUPINE_ROOT_POS = (0.0, 0.0, 0.085)
SUPINE_ROOT_QUAT = (math.cos(_SUPINE_PITCH / 2), 0.0, math.sin(_SUPINE_PITCH / 2), 0.0)

# Statically balanced crouches within the hardware joint limits: torso upright, feet
# flat (hip + knee + ankle pitch = 0) and the CoM over the feet, found by a kinematic
# search. With these limits (hip flexion <= 1.25 rad) the lowest balanced crouch has
# the base at ~0.19 m; from a sit on the ground (base ~0.06 m) the robot cannot reach
# it through balanced poses, so these starts let the squat-to-stand part be learned.
# Root heights are the kinematic values plus 5 mm clearance.
_SQUAT_POSES: tuple[tuple[float, float, float, float], ...] = (
  # (root z, hip pitch, knee, ankle pitch)
  (0.199, -1.125, 1.625, -0.5),
  (0.241, -0.8, 1.3, -0.5),
  (0.282, -0.5, 0.925, -0.425),
)
SQUAT_RESETS: tuple[dict, ...] = tuple(
  {
    "root_pos": (0.0, 0.0, z),
    "root_quat": (1.0, 0.0, 0.0, 0.0),
    "joint_pos": {
      r".*_hip_pitch_joint": hip,
      r".*_hip_roll_joint": 0.0,
      r".*_thigh_joint": 0.0,
      r".*_calf_joint": knee,
      r".*_ankle_pitch_joint": ankle,
      r".*_ankle_roll_joint": 0.0,
    },
  }
  for z, hip, knee, ankle in _SQUAT_POSES
)
# Share of resets that start crouched instead of lying on the back.
_SQUAT_RESET_PROBABILITY = 0.4


def minipi_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create HighTorque Mini-Pi getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.scene.entities = {"robot": get_minipi_robot_cfg()}

  # Mini-Pi has no armature, so the generic 5 ms timestep chatters (as in the Mini-Pi
  # velocity task). 2 ms x 10 keeps the 50 Hz policy rate.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10

  # Self-collision sensor (history_length matches decimation).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  # Very weak (counts up to 10 substeps): without arms, Mini-Pi may need torso, shin
  # and knee contacts while getting up.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.01,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Height weighs more than orientation so that leaning forward out of a sit into a
  # squat (losing some orientation reward) still pays. Scaled by uprightness: with
  # weight 2 and no scaling, the policy lifted the base 0.16 m in an upside-down
  # shoulder stand.
  cfg.rewards["torso_height"].weight = 2.0
  cfg.rewards["torso_height"].params["scale_by_uprightness"] = True
  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Per-joint posture std (thigh_joint is the hip yaw, calf_joint is the knee). Wide
  # enough that a policy standing in a crouch still gets a gradient toward the default
  # pose (at 0.08-0.2 the reward was ~exp(-30) = 0). hip_roll is nearly free so a wide
  # stance is not penalized.
  cfg.rewards["posture"].params["std"] = {
    r".*_hip_roll_joint": 0.8,
    r".*_thigh_joint": 0.3,
    r".*_hip_pitch_joint": 0.45,
    r".*_calf_joint": 0.6,
    r".*_ankle_pitch_joint": 0.6,
    r".*_ankle_roll_joint": 0.45,
  }
  # Sitting upright on the ground must not earn posture reward.
  cfg.rewards["posture"].params["min_height"] = 0.8 * _TORSO_HEIGHT

  # Once the torso is up (sitting), pull the feet back under the hips: the step from a
  # sit into a squat. Stays active while standing (feet under hips is the stance).
  cfg.rewards["feet_under_base"] = RewardTermCfg(
    func=mdp.feet_under_base,
    weight=1.0,
    params={
      "std": 0.08,
      "forward_offset": _FOOT_FORWARD_OFFSET,
      "upright_gravity_z": -0.7,
      "asset_cfg": SceneEntityCfg("robot", body_names=(r".*_ankle_roll_link",)),
    },
  )

  # Penalize spinning from the first episode, including the low preparation phase.
  # L1 charges for angular travel even for a slow turn; L2 discourages fast spins.
  cfg.rewards["base_yaw_rate"] = RewardTermCfg(func=mdp.base_yaw_rate_l2, weight=-0.1)
  cfg.rewards["base_yaw_travel"] = RewardTermCfg(func=mdp.base_yaw_rate_l1, weight=-0.5)
  # No stepping/shuffling once upright.
  cfg.rewards["upright_base_drift"] = RewardTermCfg(
    func=mdp.upright_base_lin_vel_xy_l2, weight=-1.0
  )
  cfg.rewards["upward_launch"] = RewardTermCfg(
    func=mdp.upward_speed_when_low,
    weight=-4.0,
    # Only while lying: rising out of a squat needs speed.
    params={"max_height": 0.12, "max_speed": 0.3},
  )
  # Encourage opening the legs (hip roll to its outward limit) while still lying, before
  # leaning the torso up. Off once the torso is up, so a wide sit is not a resting
  # point and the legs are free to close for the squat.
  cfg.rewards["hip_roll_open"] = RewardTermCfg(
    func=mdp.hip_roll_open_when_low,
    weight=0.3,
    params={
      "max_height": 0.2,
      "lying_gravity_z": 0.5,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_roll_joint",)),
    },
  )

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.0

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  cfg.events["geom_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.5, 1.2),
      "shared_random": True,
    },
  )

  # Supine / crouch resets instead of the random fallen/standing reset.
  del cfg.events["reset_fallen_or_standing"]
  # Mostly the supine start; the rest start in one of the balanced crouches.
  supine = {
    "root_pos": SUPINE_ROOT_POS,
    "root_quat": SUPINE_ROOT_QUAT,
    "joint_pos": SUPINE_JOINT_POS,
  }
  squat_prob = _SQUAT_RESET_PROBABILITY / len(SQUAT_RESETS)
  cfg.events["reset_supine_or_squat"] = EventTermCfg(
    func=mdp.reset_pose_mixture,
    mode="reset",
    params={
      "poses": (supine, *SQUAT_RESETS),
      "probabilities": (
        1.0 - _SQUAT_RESET_PROBABILITY,
        *(squat_prob,) * len(SQUAT_RESETS),
      ),
      "joint_pos_noise": 0.025,
    },
  )

  # The robot starts at rest on the ground, so only a short settle is needed.
  assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
  cfg.actions["joint_pos"].settle_steps = 10  # 0.2s at 50Hz action rate.
  # Relative targets: target = current joint pos + 0.25 * action. Smaller steps than
  # the generic 0.6 for gentler, hardware-friendly motion.
  cfg.actions["joint_pos"].scale = 0.25
  # Hardware joint limits: never command a joint past its range (into the hard stop).
  cfg.actions["joint_pos"].clip_target_to_limits = True
  cfg.terminations["energy"].params["settle_steps"] = 10

  # Slow, unhurried getup: 20 s episodes (1000 steps), and joint velocity / action rate
  # penalties that stay small while the getup is discovered (a random policy flails, and
  # full-strength penalties make lying still the best option), then ramp up. Steps are
  # env steps (24 per iteration). No energy threshold curriculum (T1's power limits
  # scaled by mass are guesses, not measured Mini-Pi limits), so the energy termination
  # stays at its inert inf threshold.
  cfg.episode_length_s = 20.0
  cfg.curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 400 * 24, "weight": -0.03},
          {"step": 700 * 24, "weight": -0.05},
        ],
      },
    ),
    "joint_vel_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_l2",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 400 * 24, "weight": -0.002},
          {"step": 700 * 24, "weight": -0.005},
          {"step": 1000 * 24, "weight": -0.01},
        ],
      },
    ),
  }

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
