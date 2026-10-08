"""Mini-Pi plant for the FTSR reference port (experiment pd16_noslew).

- Model: the unchanged ``cl_pai.xml`` (6.94 kg, vendor-URDF joint ranges, armature 0).
- PD gains: the deployed RL gains of the vendor controller and every mini_pi_fsm RL
  package (``HARDWARE_CONSTRAINT``).
- Actuator: MuJoCo's native ``<position>`` actuator, i.e. the motor firmware law

      tau = clip(kp (q* - q) - kd qdot, -TAU_CAP, TAU_CAP),   TAU_CAP = 16 Nm

  The clip is MuJoCo's actuator ``forcerange`` (``effort_limit``). kd is integrated
  implicitly (implicitfast). There is no torque-speed derating: the earlier
  H-conservative / H-loose envelope hypotheses were removed in this experiment
  (user decision 2026-10-08). The passive state (kp 0, kd 1) is written by the
  action term (``mdp/actions.py``). No armature is added.
"""

from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from minipi_getup.asset_zoo.robots.hightorque_minipi.minipi_constants import (
  FULL_COLLISION,
  HOME_KEYFRAME,
  get_spec,
)

##
# Joint order, gains and ranges (policy order = MuJoCo order = mjlab47 deploy order).
##

JOINT_SUFFIXES = (
  "hip_pitch_joint",
  "hip_roll_joint",
  "thigh_joint",  # hip yaw
  "calf_joint",  # knee
  "ankle_pitch_joint",
  "ankle_roll_joint",
)
JOINT_NAMES = tuple(f"{side}_{s}" for side in ("r", "l") for s in JOINT_SUFFIXES)

# Vendor RL controller running_kp/kd = mini_pi_fsm RL packages (HARDWARE_CONSTRAINT).
KP = {
  "hip_pitch": 60.0,
  "hip_roll": 40.0,
  "thigh": 20.0,
  "calf": 60.0,
  "ankle_pitch": 30.0,
  "ankle_roll": 10.0,
}
KD = {
  "hip_pitch": 2.4,
  "hip_roll": 0.8,
  "thigh": 0.4,
  "calf": 2.8,
  "ankle_pitch": 1.6,
  "ankle_roll": 0.3,
}
# mini_pi_fsm Passive state: kp 0, kd 1 on every joint (= HtdwMotor::protectMotor()).
PASSIVE_KD = 1.0

# Physical ranges from the XML (identical to the vendor URDF), policy order.
JOINT_RANGES = {
  "r_hip_pitch_joint": (-1.25, 1.75),
  "r_hip_roll_joint": (-0.5, 0.12),
  "r_thigh_joint": (-0.6, 0.3),
  "r_calf_joint": (-0.65, 1.65),
  "r_ankle_pitch_joint": (-0.5, 1.3),
  "r_ankle_roll_joint": (-0.3, 0.8),
  "l_hip_pitch_joint": (-1.25, 1.75),
  "l_hip_roll_joint": (-0.12, 0.5),
  "l_thigh_joint": (-0.3, 0.6),
  "l_calf_joint": (-0.65, 1.65),
  "l_ankle_pitch_joint": (-0.5, 1.3),
  "l_ankle_roll_joint": (-0.8, 0.3),
}

##
# Geometry (measured on the compiled model; re-checked by validate.py).
##

MINIPI_MASS = 6.94  # kg
GRAVITY = 9.81
MINIPI_WEIGHT = MINIPI_MASS * GRAVITY  # 68.1 N
STANCE_HEIGHT = 0.345  # base_link origin above the ground, all joints 0
# Reference robot (JiaRan, getup_gym release): 27.669 kg, 0.75 m stance.
REF_MASS = 27.669
REF_STANCE = 0.75
LENGTH_RATIO = STANCE_HEIGHT / REF_STANCE  # 0.46

##
# Torque limits.
##

TAU_CAP = 16.0  # Nm, hard actuator force limit (MuJoCo forcerange)
TAU_RATED_REPORT = 6.0  # Nm, reporting threshold only (time above), not enforced


def get_robot_cfg() -> EntityCfg:
  """Mini-Pi with the deployed PD gains on native position actuators.

  ``effort_limit`` = TAU_CAP sets the static actuator ``forcerange`` (+-16 Nm).
  """
  actuators = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(f".*_{joint}_joint",),
      stiffness=KP[joint],
      damping=KD[joint],
      effort_limit=TAU_CAP,
    )
    for joint in KP
  )
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=actuators),
  )
