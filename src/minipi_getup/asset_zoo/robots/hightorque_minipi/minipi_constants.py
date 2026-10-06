"""HighTorque Mini-Pi constants.

Model, joint names, PD gains and home keyframe are taken from the Mini-Pi velocity
project (minipi_velocity/minipi/minipi_constants.py and xmls/cl_pai.xml).
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MINIPI_XML: Path = Path(__file__).parent / "xmls" / "cl_pai.xml"
assert MINIPI_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MINIPI_XML))


##
# Actuator config.
##

# Physical torque cap for every joint. The vendor files give no per-joint motor specs
# (the source config left the effort limit unset, i.e. unlimited), so a single
# conservative cap is applied to all 12 actuators.
MINIPI_EFFORT_LIMIT = 16.0

# PD gains from HighTorque's RL deployment configs (sim2real walk/dreamwaq.yaml and
# clpai_12dof_0905 devel_config/config.yaml). Armature and frictionloss are left unset.
# thigh_joint is the hip yaw (z axis), calf_joint is the knee.
MINIPI_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=60.0,
  damping=2.4,
  effort_limit=MINIPI_EFFORT_LIMIT,
)
MINIPI_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=40.0,
  damping=0.8,
  effort_limit=MINIPI_EFFORT_LIMIT,
)
MINIPI_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thigh_joint",),
  stiffness=20.0,
  damping=0.4,
  effort_limit=MINIPI_EFFORT_LIMIT,
)
MINIPI_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=60.0,
  damping=2.8,
  effort_limit=MINIPI_EFFORT_LIMIT,
)
MINIPI_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=1.6,
  effort_limit=MINIPI_EFFORT_LIMIT,
)
MINIPI_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=10.0,
  damping=0.3,
  effort_limit=MINIPI_EFFORT_LIMIT,
)

##
# Keyframes.
##

# The MJCF bakes the nominal knee-bent stance into the body frames, so zero joint
# angles is the standing pose. Base height matches HighTorque's pai_12dof.xml.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.347),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[lr]_foot[1-5]_collision$"

# All collisions enabled, including self collisions. Feet get priority and friction
# 0.6 as in the velocity task; the remaining geoms get condim 3 (instead of the
# velocity task's frictionless condim 1) because the torso, shins and knees push
# against the ground while getting up.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority={_foot_regex: 1, ".*": 0},
  friction=(0.6,),
)

##
# Final config.
##

MINIPI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MINIPI_ACTUATOR_HIP_PITCH,
    MINIPI_ACTUATOR_HIP_ROLL,
    MINIPI_ACTUATOR_THIGH,
    MINIPI_ACTUATOR_CALF,
    MINIPI_ACTUATOR_ANKLE_PITCH,
    MINIPI_ACTUATOR_ANKLE_ROLL,
  ),
)


def get_minipi_robot_cfg() -> EntityCfg:
  """Get a fresh Mini-Pi robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MINIPI_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_minipi_robot_cfg())

  viewer.launch(robot.spec.compile())
