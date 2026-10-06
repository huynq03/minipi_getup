from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import minipi_getup_env_cfg
from .force_guided_env_cfgs import (
  minipi_getup_force_guided_env_cfg,
  minipi_getup_force_guided_released_env_cfg,
  minipi_getup_force_guided_safe_env_cfg,
  minipi_getup_force_guided_stand_env_cfg,
)
from .rl_cfg import minipi_getup_ppo_runner_cfg, minipi_getup_safe_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-MiniPi",
  env_cfg=minipi_getup_env_cfg(),
  play_env_cfg=minipi_getup_env_cfg(play=True),
  rl_cfg=minipi_getup_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-MiniPi-ForceGuided",
  env_cfg=minipi_getup_force_guided_env_cfg(),
  play_env_cfg=minipi_getup_force_guided_env_cfg(play=True),
  rl_cfg=minipi_getup_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-MiniPi-ForceGuidedReleased",
  env_cfg=minipi_getup_force_guided_released_env_cfg(),
  play_env_cfg=minipi_getup_force_guided_released_env_cfg(play=True),
  rl_cfg=minipi_getup_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-MiniPi-ForceGuidedStand",
  env_cfg=minipi_getup_force_guided_stand_env_cfg(),
  play_env_cfg=minipi_getup_force_guided_stand_env_cfg(play=True),
  rl_cfg=minipi_getup_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Getup-Flat-MiniPi-ForceGuidedSafe",
  env_cfg=minipi_getup_force_guided_safe_env_cfg(),
  play_env_cfg=minipi_getup_force_guided_safe_env_cfg(play=True),
  rl_cfg=minipi_getup_safe_ppo_runner_cfg(),
)
