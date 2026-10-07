"""Mini-Pi robot cfg for FTSR: the shared asset with an operational torque envelope.

The physical/emergency limit of the Mini-Pi motors is 16 Nm
(``MINIPI_EFFORT_LIMIT``, used by the baseline tasks). FTSR trains inside a stricter
operational envelope: the actuators saturate at ``OPERATIONAL_TORQUE_LIMIT``. 16 Nm
stays a hardware ceiling, not something the policy may learn to rely on. The shared
asset definition isn't changed.
"""

import math
from dataclasses import replace

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityCfg

from minipi_getup.asset_zoo.robots.hightorque_minipi.minipi_constants import (
  MINIPI_EFFORT_LIMIT,
  get_minipi_robot_cfg,
)

# Operational joint torque limit (Nm) of the faithful task. Keep within 8-10 Nm.
OPERATIONAL_TORQUE_LIMIT = 9.0
# Get-up-first variants may use more: the user raised the envelope to 12-13 Nm on
# 2026-10-07 so that a slow get-up is feasible. 16 Nm stays the hardware ceiling.
MAX_OPERATIONAL_TORQUE_LIMIT = 13.0
assert OPERATIONAL_TORQUE_LIMIT <= MAX_OPERATIONAL_TORQUE_LIMIT < MINIPI_EFFORT_LIMIT

# Measured on the compiled model (docs/FTSR_REPRODUCTION_AUDIT.md, Sec. 2).
MINIPI_MASS = 6.94  # kg
MINIPI_WEIGHT = MINIPI_MASS * 9.81  # N
STANCE_HEIGHT = 0.345  # base_link origin above the soles, all joints at 0
STANCE_COM_HEIGHT = 0.223
# Lateral distance between the ankle_roll_link origins in the stance.
STANCE_FEET_WIDTH = 0.16


def get_ftsr_robot_cfg(torque_limit: float = OPERATIONAL_TORQUE_LIMIT) -> EntityCfg:
  assert torque_limit <= MAX_OPERATIONAL_TORQUE_LIMIT, torque_limit
  cfg = get_minipi_robot_cfg()
  assert cfg.articulation is not None
  cfg.articulation = replace(
    cfg.articulation,
    actuators=tuple(
      replace(a, effort_limit=torque_limit) for a in cfg.articulation.actuators
    ),
  )
  return cfg


# HTDW-5036-02 joint module (Mini-Pi), datasheet values supplied by the user
# (2026-10-07; the public product page confirms the module name and the 16 Nm robot
# maximum only): rated 6 Nm at 50 rpm, locked-rotor 21 Nm, no-load 75 rpm, 36:1.
HTDW5036_STALL_TORQUE = 21.0  # Nm
HTDW5036_NO_LOAD_SPEED = 75.0 * 2.0 * math.pi / 60.0  # 7.85 rad/s at the joint
# Reflected rotor inertia (kg m^2). Not in the datasheet; ~1e-5 kg m^2 for a 50 mm
# rotor x 36^2 gives ~0.013. Also required numerically: the torch PD law is explicit,
# and without armature the light links chatter at up to ~40 rad/s while holding still
# at the 2 ms timestep (0.005 already stabilizes it).
HTDW5036_ARMATURE = 0.01


def get_ftsr_dc_robot_cfg(effort_limit: float = MINIPI_EFFORT_LIMIT) -> EntityCfg:
  """Mini-Pi with a DC motor torque-speed curve instead of a flat torque cap.

  tau_max(qd) = min(effort_limit, 21 (1 - |qd| / 7.85)), the linear curve through the
  datasheet's stall torque and no-load speed (it gives 7.0 Nm at the rated 5.24 rad/s,
  close to the rated 6 Nm). The PD law (same kp/kd as the builtin actuators) runs in
  torch every physics step. ``effort_limit`` defaults to the 16 Nm the real motors
  allow: the deployment runs the motors' own PD with no torque clamp.
  """
  cfg = get_minipi_robot_cfg()
  assert cfg.articulation is not None
  cfg.articulation = replace(
    cfg.articulation,
    actuators=tuple(
      DcMotorActuatorCfg(
        target_names_expr=a.target_names_expr,
        stiffness=a.stiffness,
        damping=a.damping,
        effort_limit=effort_limit,
        saturation_effort=HTDW5036_STALL_TORQUE,
        velocity_limit=HTDW5036_NO_LOAD_SPEED,
        armature=HTDW5036_ARMATURE,
      )
      for a in cfg.articulation.actuators
    ),
  )
  return cfg
