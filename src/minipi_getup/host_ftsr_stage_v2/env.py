"""v2 env: v1 StageEnv with StageLogicV2, the joint-margin feature/reward and monitors."""

import torch

from minipi_getup.host_ftsr_stage.env import StageEnv

from .rewards import StageLogicV2


class StageEnvV2(StageEnv):
  def __init__(self, cfg, num_envs, device="cuda:0"):
    super().__init__(cfg, num_envs, device)
    self.stage_logic = StageLogicV2(num_envs, device, self.dt, self.geometry)
    self.stage_settings = self.stage_logic.cfg
    z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
    self.margin_sum, self.near_steps, self.policy_steps = z(), z(), z()
    self.stage3_entered = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def measure(self):
    f = super().measure()
    m = self.stage_settings.joint_margin_rad
    lo, hi = self.physical_limits.T
    q = self.dof_pos
    f["margin_dist"] = (lo + m - q).clamp(min=0) + (q - (hi - m)).clamp(min=0)
    return f

  def _reward_joint_margin(self):
    return self.stage_logic.last["joint_margin"]

  def compute_reward(self):
    super().compute_reward()
    last = self.stage_logic.last
    active = (self.real_episode_length_buf > self.unactuated_time).float()
    self.policy_steps.add_(active)
    self.margin_sum.add_(last["joint_margin"])
    self.near_steps.add_((self.features["margin_dist"] > 0).any(-1).float() * active)
    self.stage3_entered.logical_or_(self.stage_logic.stage >= 2)

  def _log_reset_metrics(self, ids):
    super()._log_reset_metrics(ids)
    if not hasattr(self, "margin_sum"):
      return
    s = self.reset_stats
    steps = self.policy_steps[ids].clamp(min=1)
    over = self.max_overshoot[ids]
    s.update(
      stage3_entry_fraction=self.stage3_entered[ids].float().mean(),
      stage3_dwell_time=self.stage_logic.time_in_stage[ids, 2].mean(),
      joint_overshoot_max=over.max(),
      joint_overshoot_p95=over.quantile(0.95),
      joint_margin_reward=(self.margin_sum[ids] / steps).mean(),
      near_joint_limit_fraction=(self.near_steps[ids] / steps).mean(),
      peak_torque_p95=self.peak_torque[ids].max(-1).values.quantile(0.95),
      peak_joint_velocity_p95=self.peak_joint_vel[ids].max(-1).values.quantile(0.95),
    )

  def reset_idx(self, ids):
    super().reset_idx(ids)
    if len(ids) and hasattr(self, "margin_sum"):
      for t in (
        self.margin_sum,
        self.near_steps,
        self.policy_steps,
        self.stage3_entered,
      ):
        t[ids] = 0
