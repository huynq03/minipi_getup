"""Numerical safety guards isolated to the safe-recovery experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def install_action_std_guard(
  algorithm: Any, minimum: float = 0.05, maximum: float = 1.0
) -> None:
  """Keep HoST's trainable action std inside a finite exploration envelope.

  The original PPO update, entropy objective, and adaptive learning-rate schedule stay
  untouched.  Clamping happens after each complete update and before the next rollout,
  which prevents a growing std from producing destructive position-target samples.
  """
  if not 0.0 < minimum <= maximum:
    raise ValueError("action std bounds must satisfy 0 < minimum <= maximum")

  actor = algorithm.actor_critic
  original_update: Callable[..., Any] = algorithm.update

  def clamp() -> None:
    with torch.no_grad():
      actor.std.clamp_(minimum, maximum)

  def guarded_update(*args: Any, **kwargs: Any) -> Any:
    result = original_update(*args, **kwargs)
    clamp()
    return result

  clamp()
  algorithm.update = guarded_update
  algorithm.action_std_bounds = (minimum, maximum)
