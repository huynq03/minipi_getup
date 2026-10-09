"""Register the FTSR reference tasks."""

from mjlab.tasks.registry import register_mjlab_task

from minipi_getup.ftsr_ref.config.env_cfg import recovery_env_cfg, walk_env_cfg
from minipi_getup.ftsr_ref.config.rl_cfg import (
  recovery_limiter_runner_cfg,
  recovery_runner_cfg,
  walk_runner_cfg,
)
from minipi_getup.ftsr_ref.rl.runner import FtsrRunner

register_mjlab_task(
  task_id="Mjlab-FTSR-Ref-MiniPi-Walk",
  env_cfg=walk_env_cfg(),
  play_env_cfg=walk_env_cfg(play=True),
  rl_cfg=walk_runner_cfg(),
  runner_cls=FtsrRunner,
)

register_mjlab_task(
  task_id="Mjlab-FTSR-Ref-MiniPi-Recovery",
  env_cfg=recovery_env_cfg(),
  play_env_cfg=recovery_env_cfg(play=True),
  rl_cfg=recovery_runner_cfg(),
  runner_cls=FtsrRunner,
)

# Recovery v2 semantics (stateless stages that may regress), to resume / evaluate
# the v2 run under the code it was trained with. Otherwise identical.
register_mjlab_task(
  task_id="Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless",
  env_cfg=recovery_env_cfg(monotonic=False),
  play_env_cfg=recovery_env_cfg(play=True, monotonic=False),
  rl_cfg=recovery_runner_cfg(),
  runner_cls=FtsrRunner,
)

# Limiter fine-tune: the stateless task with a 0.3 rad / policy-step PD-target rate
# limiter in both the training and the play cfg (evaluation applies it once, via the
# cfg). Otherwise identical to Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless.
LIMIT_TASK = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3"
register_mjlab_task(
  task_id=LIMIT_TASK,
  env_cfg=recovery_env_cfg(monotonic=False, target_rate_limit=0.3),
  play_env_cfg=recovery_env_cfg(play=True, monotonic=False, target_rate_limit=0.3),
  rl_cfg=recovery_limiter_runner_cfg(LIMIT_TASK),
  runner_cls=FtsrRunner,
)

# Zero-shot viewing / evaluation of a tighter limiter (0.1 rad / policy step). Same
# runner cfg as the 0.3 task (periodic evaluation of itself) if it is ever trained.
LIMIT_TASK_0P1 = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p1"
register_mjlab_task(
  task_id=LIMIT_TASK_0P1,
  env_cfg=recovery_env_cfg(monotonic=False, target_rate_limit=0.1),
  play_env_cfg=recovery_env_cfg(play=True, monotonic=False, target_rate_limit=0.1),
  rl_cfg=recovery_limiter_runner_cfg(LIMIT_TASK_0P1),
  runner_cls=FtsrRunner,
)
