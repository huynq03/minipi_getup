"""Height-progressive stage-wise rewards (FTSR paper Sec. II-D, Alg. 1 lines 9-11).

Three reward sets share one reward manager: r_u (upper-body erection), r_s (standing
up) and r_w (walking). Each stage sets the target base height ``h_cmd_i``:

    S_1 = {i | h_i > h1},  S_2 = {i | h_i > h2}
    stage = r_w  if |S_2| > 2/3 N
            r_s  if |S_1| > 2/3 N
            r_u  otherwise
    h_cmd = (h1, h2, h3)[stage]

The paper's Alg. 1 writes this as ``p = 2 I(|S_1| > 2N/3) + I(|S_2| > 2N/3)`` with
``R = [r^u, r^w, r^s]``. Taken literally, that maps "S_1 only" to r^s but "both" to
p = 3, which has no reward set. The text (Sec. II-D) is unambiguous, so it is followed.

Like Alg. 1 (where the stage is recomputed every iteration) and the released code
(recomputed every env step), the stage isn't latched: if the population falls back
below a threshold, the earlier stage returns. ``|S|/N`` is averaged over one PPO
iteration's env steps and the stage is re-decided once per iteration, so a single
step with many fresh resets cannot flip it.

A stage applies its weights to every listed reward term (0 = inactive, the paper's
"-"). The target height goes to ``env.extras[H_CMD_KEY]`` for the height reward and
for the Eq. 4 assistance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from minipi_getup.ftsr.mdp.assistance import H_CMD_KEY

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STAGE_KEY = "ftsr_stage"
STAGE_NAMES = ("upper_body", "stand_up", "walk")

_TORSO = SceneEntityCfg("robot", body_names=("base_link",))


class ftsr_stage_manager:
  """Step event that decides the reward stage from population height statistics.

  Params:
    heights: (h1, h2, h3) target base heights of the three stages.
    weights: {reward term name: (w_u, w_s, w_w)}.
    fraction: population share that must be above a threshold (paper: 2/3).
    steps_per_iteration: env steps between decisions (rsl-rl ``num_steps_per_env``).
    fixed_stage: force one stage (e.g. 2 for walking-only pretraining).
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._heights: tuple[float, float, float] = tuple(params["heights"])
    self._weights: dict[str, tuple[float, float, float]] = params["weights"]
    self._fixed: int | None = params.get("fixed_stage")
    self._sum_s1 = 0.0
    self._sum_s2 = 0.0
    self._count = 0
    self._last_s1 = 0.0
    self._last_s2 = 0.0
    self._transitions: list[tuple[int, int]] = []
    self.stage = -1
    # The event manager is built before the reward manager, so the first stage is
    # applied on the first call. The config's own reward weights must therefore
    # already be the first stage's (``config/stage_rewards.py`` does that).
    env.extras[H_CMD_KEY] = self._heights[self._fixed or 0]

  def _apply(self, env: ManagerBasedRlEnv, stage: int) -> None:
    if stage != self.stage:
      self._transitions.append((env.common_step_counter, stage))
    self.stage = stage
    env.extras[H_CMD_KEY] = self._heights[stage]
    env.extras[STAGE_KEY] = stage
    for name, per_stage in self._weights.items():
      env.reward_manager.get_term_cfg(name).weight = per_stage[stage]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    heights: tuple[float, float, float],
    weights: dict,
    fraction: float = 2.0 / 3.0,
    steps_per_iteration: int = 24,
    fixed_stage: int | None = None,
    asset_cfg: SceneEntityCfg = _TORSO,
  ) -> None:
    del env_ids, weights, fixed_stage
    if self.stage < 0:
      self._apply(env, self._fixed if self._fixed is not None else 0)
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]
    self._sum_s1 += (h > heights[0]).float().mean().item()
    self._sum_s2 += (h > heights[1]).float().mean().item()
    self._count += 1
    if self._count >= steps_per_iteration:
      self._last_s1 = self._sum_s1 / self._count
      self._last_s2 = self._sum_s2 / self._count
      self._sum_s1 = self._sum_s2 = 0.0
      self._count = 0
      if self._fixed is None:
        if self._last_s2 > fraction:
          stage = 2
        elif self._last_s1 > fraction:
          stage = 1
        else:
          stage = 0
        self._apply(env, stage)

    log = env.extras.setdefault("log", {})
    log["Stage/stage"] = float(self.stage)
    log["Stage/h_cmd"] = self._heights[self.stage]
    log["Stage/frac_above_h1"] = self._last_s1
    log["Stage/frac_above_h2"] = self._last_s2

  @property
  def transitions(self) -> list[tuple[int, int]]:
    """(env step, new stage) for every stage change since construction."""
    return self._transitions
