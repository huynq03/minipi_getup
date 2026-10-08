"""Register the FTSR reference tasks."""

from mjlab.tasks.registry import register_mjlab_task

from minipi_getup.ftsr_ref.config.env_cfg import recovery_env_cfg, walk_env_cfg
from minipi_getup.ftsr_ref.config.rl_cfg import recovery_runner_cfg, walk_runner_cfg
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
