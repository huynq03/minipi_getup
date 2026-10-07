"""Register the FTSR Mini-Pi tasks."""

from mjlab.tasks.registry import register_mjlab_task

from minipi_getup.ftsr.config.env_cfg import ftsr_env_cfg, ftsr_walk_env_cfg
from minipi_getup.ftsr.config.rl_cfg import ftsr_runner_cfg, ftsr_walk_runner_cfg
from minipi_getup.ftsr.config.tuning import (
  DEPLOY_ACTION_STEP,
  DEPLOY_ASSIST_END_ITERATION,
  DEPLOY_FIXED_STAGE,
  DEPLOY_MOTOR_CAP,
  DEPLOY_TORQUE_ENVELOPE,
  GENTLE_TARGET_STEP,
  GENTLE_TORQUE_LIMIT,
  GETUP_DEPLOY,
  GETUP_SOFT,
  GETUP_SOFT_V2,
  SMOOTH_V1,
  SMOOTH_V2,
)
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

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-GetupSoft",
  env_cfg=ftsr_env_cfg(stage_weight_overrides=GETUP_SOFT, stand_only=True),
  play_env_cfg=ftsr_env_cfg(
    play=True, stage_weight_overrides=GETUP_SOFT, stand_only=True
  ),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-GetupSoft2",
  env_cfg=ftsr_env_cfg(stage_weight_overrides=GETUP_SOFT_V2, stand_only=True),
  play_env_cfg=ftsr_env_cfg(
    play=True, stage_weight_overrides=GETUP_SOFT_V2, stand_only=True
  ),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)

register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-GetupGentle",
  env_cfg=ftsr_env_cfg(
    stand_only=True,
    torque_limit=GENTLE_TORQUE_LIMIT,
    max_target_step=GENTLE_TARGET_STEP,
  ),
  play_env_cfg=ftsr_env_cfg(
    play=True,
    stand_only=True,
    torque_limit=GENTLE_TORQUE_LIMIT,
    max_target_step=GENTLE_TARGET_STEP,
  ),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)


def _getup_deploy_cfg(play: bool):
  return ftsr_env_cfg(
    play=play,
    stage_weight_overrides=GETUP_DEPLOY,
    stand_only=True,
    torque_limit=DEPLOY_TORQUE_ENVELOPE,
    dc_motor_effort_limit=DEPLOY_MOTOR_CAP,
    max_action_step=DEPLOY_ACTION_STEP,
    assist_end_iteration=DEPLOY_ASSIST_END_ITERATION,
    fixed_stage=DEPLOY_FIXED_STAGE,
  )


register_mjlab_task(
  task_id="Mjlab-FTSR-MiniPi-GetupDeploy",
  env_cfg=_getup_deploy_cfg(play=False),
  play_env_cfg=_getup_deploy_cfg(play=True),
  rl_cfg=ftsr_runner_cfg(),
  runner_cls=FtsrRunner,
)
