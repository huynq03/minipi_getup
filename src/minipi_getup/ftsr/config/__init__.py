"""Register the FTSR Mini-Pi tasks."""

from mjlab.tasks.registry import register_mjlab_task

from minipi_getup.ftsr.config.env_cfg import ftsr_env_cfg, ftsr_walk_env_cfg
from minipi_getup.ftsr.config.rl_cfg import ftsr_runner_cfg, ftsr_walk_runner_cfg
from minipi_getup.ftsr.config.tuning import SMOOTH_V1, SMOOTH_V2
from minipi_getup.ftsr.rl import FtsrRunner

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi",
  env_cfg=ftsr_env_cfg(),
  play_env_cfg=ftsr_env_cfg(play=True),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-Walk",
  env_cfg=ftsr_walk_env_cfg(),
  play_env_cfg=ftsr_walk_env_cfg(play=True),
  rl_cfg=ftsr_walk_runner_cfg(),
  runner_cls=FtsrRunner,
)

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-TuneSmooth1",
  env_cfg=ftsr_env_cfg(stage_weight_overrides=SMOOTH_V1),
  play_env_cfg=ftsr_env_cfg(play=True, stage_weight_overrides=SMOOTH_V1),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-TuneSmooth2",
  env_cfg=ftsr_env_cfg(stage_weight_overrides=SMOOTH_V2),
  play_env_cfg=ftsr_env_cfg(play=True, stage_weight_overrides=SMOOTH_V2),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)
