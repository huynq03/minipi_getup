"""Velocity command that works with a zero-width command range in the viser GUI."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

# viser's "Max <axis>" slider starts at the range maximum and needs it in [0.1, 10].
_MIN_GUI_RANGE = 0.1


class FtsrVelocityCommand(UniformVelocityCommand):
  """``UniformVelocityCommand`` whose GUI accepts axes with a zero range.

  The FTSR commands keep lin_vel_y at (0, 0), which makes mjlab's joystick GUI
  fail to build. Sampling still uses the configured ranges. Only the joystick
  limits are widened to the slider minimum.
  """

  def create_gui(self, *args, **kwargs) -> None:
    ranges = self.cfg.ranges
    gui_ranges = dataclasses.replace(
      ranges,
      **{
        axis: (min(lo, -_MIN_GUI_RANGE), max(hi, _MIN_GUI_RANGE))
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        for lo, hi in [getattr(ranges, axis)]
      },
    )
    self.cfg.ranges = gui_ranges
    try:
      super().create_gui(*args, **kwargs)
    finally:
      self.cfg.ranges = ranges


@dataclass(kw_only=True)
class FtsrVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env) -> FtsrVelocityCommand:
    return FtsrVelocityCommand(self, env)
