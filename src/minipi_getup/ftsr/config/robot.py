"""Mini-Pi robot cfg for FTSR: the shared asset with an operational torque envelope.

The physical/emergency limit of the Mini-Pi motors is 16 Nm
(``MINIPI_EFFORT_LIMIT``, used by the baseline tasks). FTSR trains inside a stricter
operational envelope: the actuators saturate at ``OPERATIONAL_TORQUE_LIMIT``. 16 Nm
stays a hardware ceiling, not something the policy may learn to rely on. The shared
asset definition isn't changed.
"""

from dataclasses import replace

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
