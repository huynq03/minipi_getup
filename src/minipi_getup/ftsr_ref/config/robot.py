"""Mini-Pi plant for the FTSR reference port (docs/MINIPI_PHYSICAL_MODEL_V2.md).

Everything here is a Mini-Pi invariant or an explicitly labelled hypothesis:

- Model: the unchanged ``cl_pai.xml`` (6.94 kg, vendor-URDF joint ranges, armature 0).
- PD gains: the deployed RL gains of the vendor controller and every mini_pi_fsm RL
  package (``HARDWARE_CONSTRAINT``). The 0.7 kp factor of the old get-up task was a
  simulation-only choice and is dropped.
- Actuator: MuJoCo's native ``<position>`` actuator (implicit in kd, stable at 2 ms
  without armature), i.e. the motor firmware law ``tau = kp (q* - q) - kd qdot``,
  force-limited at tau_cap. Every physics step the action term selects the active
  piece of the motor envelope below and writes it as the actuator's affine law
  (``mdp/actions.py``), keeping the velocity slope implicit. No armature is added
  (the vendor URDF has none; the reflected rotor inertia of the installed motor is
  unknown, see docs/MINIPI_PHYSICAL_MODEL_V2.md).
- Motor envelope: a linear four-quadrant DC curve, the same law as mjlab's
  ``dc_motor_clip``::

      tau_max(qd) = clamp(tau_stall (1 - qd / omega0), max=tau_cap)
      tau_min(qd) = clamp(tau_stall (-1 - qd / omega0), min=-tau_cap)

  Motoring torque falls to zero at the no-load speed; braking torque is only capped.
  The installed HTDW-5047-36-NE envelope is unknown, so two hypotheses are defined;
  neither is a verified motor specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
# Motor envelope hypotheses (docs/MINIPI_PHYSICAL_MODEL_V2.md, Sec. 4b).
##


@dataclass
class MotorEnvelopeCfg:
  """Linear four-quadrant torque-speed envelope at the joint (see module doc)."""

  name: str = "H-conservative"
  omega0: float = 75.0 * 2.0 * math.pi / 60.0  # rad/s, no-load speed (7.85)
  tau_stall: float = 21.0  # Nm
  tau_cap: float = 16.0  # Nm, never exceeded
  tau_rated: float = 6.0  # Nm, monitored only (time above rated)


# Nominal provisional training plant (user decision 2026-10-07). User-supplied
# HTDW-5036-02 summary: no-load 75 rpm, stall 21 Nm; 16 Nm = product-page robot max.
H_CONSERVATIVE = MotorEnvelopeCfg()
# Sensitivity / evaluation plant only: no-load speed = vendor URDF velocity limit.
H_LOOSE = MotorEnvelopeCfg(name="H-loose", omega0=21.0)
MOTOR_HYPOTHESES = {m.name: m for m in (H_CONSERVATIVE, H_LOOSE)}


def get_robot_cfg() -> EntityCfg:
  """Mini-Pi with the deployed PD gains on native position actuators.

  ``effort_limit`` = tau_cap makes the actuators force-limited; the action term then
  narrows ``forcerange`` to the motor envelope at every physics step.
  """
  actuators = tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(f".*_{joint}_joint",),
      stiffness=KP[joint],
      damping=KD[joint],
      effort_limit=H_CONSERVATIVE.tau_cap,
    )
    for joint in KP
  )
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=actuators),
  )
