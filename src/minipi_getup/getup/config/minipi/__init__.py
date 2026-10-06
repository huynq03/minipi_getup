from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import minipi_getup_env_cfg
from .rl_cfg import minipi_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-MiniPi",
  env_cfg=minipi_getup_env_cfg(),
  play_env_cfg=minipi_getup_env_cfg(play=True),
  rl_cfg=minipi_getup_ppo_runner_cfg(),
)
