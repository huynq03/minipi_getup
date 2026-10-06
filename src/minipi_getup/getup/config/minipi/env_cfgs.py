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
# the single torso body; its origin sits on the hip line. Mini-Pi has no pelvis link,
# so the pelvis height is the mean of the two hip joint centers (hip_pitch_link
# origins, 33 mm below the base origin).
_TORSO_HEIGHT = 0.344
_PELVIS_HEIGHT = 0.311

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

  # Weak: counts up to 10 substeps, so -0.04 matches T1's -0.1 over 4 substeps.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.04,
    params={"sensor_name": self_collision_cfg.name},
  )

  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  cfg.rewards["pelvis_height"] = RewardTermCfg(
    func=mdp.height_reward,
    weight=1.0,
    params={
      "desired_height": _PELVIS_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("l_hip_pitch_link", "r_hip_pitch_link")
      ),
    },
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

  # Same schedule as T1: free exploration first, then smoother/cheaper motion. Energy
  # thresholds (W) are T1's scaled by mass (6.9 kg vs ~30 kg).
  cfg.curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 600 * 24, "weight": -0.05},
          {"step": 900 * 24, "weight": -0.08},
          {"step": 1200 * 24, "weight": -0.1},
        ],
      },
    ),
    "joint_vel_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_l2",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": 900 * 24, "weight": -0.005},
          {"step": 1200 * 24, "weight": -0.008},
          {"step": 1500 * 24, "weight": -0.01},
        ],
      },
    ),
    "energy_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 900 * 24, "params": {"threshold": 700.0}},
          {"step": 1200 * 24, "params": {"threshold": 460.0}},
          {"step": 1500 * 24, "params": {"threshold": 350.0}},
          {"step": 1700 * 24, "params": {"threshold": 230.0}},
          {"step": 2200 * 24, "params": {"threshold": 160.0}},
        ],
      },
    ),
  }

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
