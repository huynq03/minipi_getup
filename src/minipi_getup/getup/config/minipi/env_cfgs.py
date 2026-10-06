"""HighTorque Mini-Pi getup environment configuration.

First stage: recover from one fixed supine (lying on the back) pose to standing.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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

  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Per-joint posture std (thigh_joint is the hip yaw, calf_joint is the knee).
  cfg.rewards["posture"].params["std"] = {
    r".*_hip_roll_joint": 0.08,
    r".*_thigh_joint": 0.08,
    r".*_hip_pitch_joint": 0.12,
    r".*_calf_joint": 0.15,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.2,
  }

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

  # Fixed supine reset instead of the random fallen/standing reset.
  del cfg.events["reset_fallen_or_standing"]
  cfg.events["reset_supine"] = EventTermCfg(
    func=mdp.reset_fixed_pose,
    mode="reset",
    params={
      "root_pos": SUPINE_ROOT_POS,
      "root_quat": SUPINE_ROOT_QUAT,
      "joint_pos": SUPINE_JOINT_POS,
      "joint_pos_noise": 0.025,
    },
  )

  # The robot starts at rest on the ground, so only a short settle is needed.
  assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
  cfg.actions["joint_pos"].settle_steps = 10  # 0.2s at 50Hz action rate.
  cfg.terminations["energy"].params["settle_steps"] = 10

  # Discovery stage: no curriculum. action_rate_l2 stays -0.01 and joint_vel_l2 stays
  # 0 so aggressive exploratory motion is not discouraged early. No energy threshold
  # curriculum either (T1's power limits scaled by mass are guesses, not measured
  # Mini-Pi limits), so the energy termination stays at its inert inf threshold.
  cfg.curriculum = {}

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
