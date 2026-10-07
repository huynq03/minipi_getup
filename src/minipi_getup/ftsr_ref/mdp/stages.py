"""Height-progressive stage-wise rewards (paper Sec. II-D).

Paper text (followed; Alg. 1's ``p`` formula maps "S_1 only" to r^s but "both" to
p = 3, which has no reward set):

    S_1 = {i | h_i > h1_cmd},  S_2 = {i | h_i > h2_cmd}
    stage = r_w if |S_2| > 2/3 N, else r_s if |S_1| > 2/3 N, else r_u
    target height h_cmd = (h1_cmd, h2_cmd, h3_cmd)[stage]

The thresholds are the earlier stages' target heights (paper: "S_1 = {i | h >
h_1^cmd}"). Decision frequency: every env step from the current population, stateless,
as the release's ``_reward_update`` (it can fall back to an earlier stage). Being a
pure function of the current state, nothing has to be checkpointed: a resumed run
recomputes the same stage from the same population.

The stage is computed once per env step, at the first reward evaluation (rewards are
the first consumer after physics), and cached by ``common_step_counter``. The Eq. 4
assistance of the next step reads the cached ``h_cmd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STAGE_KEY = "ftsr_stage"
STAGE_NAMES = ("r_u", "r_s", "r_w")


@dataclass
class StageCfg:
  heights: tuple[float, float, float]
  """(h1_cmd, h2_cmd, h3_cmd), m."""
  fraction: float = 2.0 / 3.0
  fixed_stage: int | None = None
  """Force one stage (2 = r_w for the walking initialization)."""


def decide_stage(
  h: torch.Tensor, cfg: StageCfg
) -> tuple[int, torch.Tensor, torch.Tensor]:
  """(stage, |S_1|/N, |S_2|/N) from base heights ``h`` (N,); fractions as device
  scalars. A fixed stage needs no host read; otherwise the fractions are read once
  per env step (the stage selects host-side reward weights)."""
  s1 = (h > cfg.heights[0]).float().mean()
  s2 = (h > cfg.heights[1]).float().mean()
  if cfg.fixed_stage is not None:
    return cfg.fixed_stage, s1, s2
  f1, f2 = torch.stack((s1, s2)).tolist()
  if f2 > cfg.fraction:
    return 2, s1, s2
  if f1 > cfg.fraction:
    return 1, s1, s2
  return 0, s1, s2


class StageState:
  """Cached per env step; lives in ``env.extras[STAGE_KEY]``."""

  def __init__(self, env: ManagerBasedRlEnv, cfg: StageCfg):
    self.cfg = cfg
    torso = SceneEntityCfg("robot", body_names=("base_link",))
    torso.resolve(env.scene)
    self._body = torso.body_ids[0]
    self.step = -1
    self.stage = cfg.fixed_stage if cfg.fixed_stage is not None else 0
    self.s1: torch.Tensor | float = 0.0
    self.s2: torch.Tensor | float = 0.0

  @property
  def h_cmd(self) -> float:
    return self.cfg.heights[self.stage]

  def update(self, env: ManagerBasedRlEnv) -> int:
    if self.step != env.common_step_counter:
      self.step = env.common_step_counter
      h = env.scene["robot"].data.body_link_pos_w[:, self._body, 2]
      self.stage, self.s1, self.s2 = decide_stage(h, self.cfg)
      log = env.extras.setdefault("log", {})
      log["Stage/stage"] = float(self.stage)
      log["Stage/h_cmd"] = self.h_cmd
      log["Stage/frac_above_h1"] = self.s1
      log["Stage/frac_above_h2"] = self.s2
    return self.stage


def stage_setup(env: ManagerBasedRlEnv, env_ids, stage: StageCfg) -> None:
  """Startup event: install the stage state."""
  del env_ids
  env.extras[STAGE_KEY] = StageState(env, stage)


def get_stage_state(env: ManagerBasedRlEnv) -> StageState:
  return env.extras[STAGE_KEY]
