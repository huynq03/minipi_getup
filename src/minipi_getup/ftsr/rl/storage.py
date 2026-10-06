"""Rollout storage for FTSR: rewards and constraint costs, both through GAE."""

from __future__ import annotations

import torch


class FtsrStorage:
  def __init__(
    self,
    num_steps: int,
    num_envs: int,
    dims: dict[str, int],
    num_actions: int,
    latent_dim: int,
    num_costs: int,
    device: torch.device | str,
  ):
    self.T, self.N = num_steps, num_envs
    self.device = device

    def z(*shape, dtype=torch.float):
      return torch.zeros(num_steps, num_envs, *shape, device=device, dtype=dtype)

    self.obs = {name: z(dim) for name, dim in dims.items()}
    self.z_student = z(latent_dim)
    self.actions = z(num_actions)
    self.log_prob = z()
    self.mu = z(num_actions)
    self.sigma = z(num_actions)
    self.rewards = z()
    self.costs = z(num_costs)
    self.dones = z()
    self.values = z()
    self.cost_values = z(num_costs)
    self.returns = z()
    self.cost_returns = z(num_costs)
    self.advantages = z()
    self.step = 0

  def add(self, **items: torch.Tensor | dict[str, torch.Tensor]) -> None:
    t = self.step
    for name, value in items.items():
      if name == "obs":
        assert isinstance(value, dict)
        for group, tensor in value.items():
          self.obs[group][t].copy_(tensor)
      else:
        assert isinstance(value, torch.Tensor)
        getattr(self, name)[t].copy_(value.view_as(getattr(self, name)[t]))
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  @staticmethod
  def _gae(rew, val, last_val, dones, gamma, lam):
    """Standard GAE over [T, N, k]; returns (returns, advantages)."""
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    T = rew.shape[0]
    for t in reversed(range(T)):
      next_val = last_val if t == T - 1 else val[t + 1]
      not_done = 1.0 - dones[t]
      delta = rew[t] + not_done * gamma * next_val - val[t]
      last = delta + not_done * gamma * lam * last
      adv[t] = last
    return adv + val, adv

  def compute_advantages(
    self,
    last_value: torch.Tensor,
    last_cost_value: torch.Tensor,
    gamma: float,
    lam: float,
    penalty_factors: tuple[float, ...],
    use_constraints: bool,
  ) -> dict[str, float]:
    """Paper Eq. 8 with the penalty method of Eq. 5.

    A_bar = std(A_r) - sum_i beta_i / (1 - gamma) * std(A_Ci)

    std() is per-batch standardization ("an additional standardization step is
    applied" before the gradient). The J_Ci(pi_k) term of Eq. 8 is the same for every
    sample of the batch. Its gradient contribution through the PPO ratio is zero in
    expectation, so only its value is logged. The term turns off when the rollout's
    costs are all zero: after t_tag (d_i = 0), A_bar = A_r exactly.
    """
    d = self.dones.unsqueeze(-1)
    self.returns, adv_r = self._gae(
      self.rewards.unsqueeze(-1), self.values.unsqueeze(-1), last_value, d, gamma, lam
    )
    self.returns = self.returns.squeeze(-1)
    adv_r = adv_r.squeeze(-1)
    self.cost_returns, adv_c = self._gae(
      self.costs, self.cost_values, last_cost_value, d, gamma, lam
    )

    adv = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
    info = {
      "J_C_force": float(self.costs[..., 0].mean() / (1.0 - gamma)),
      "J_C_torque": float(self.costs[..., 1].mean() / (1.0 - gamma)),
      "constraint_active": 0.0,
      "constraint_adv_std": 0.0,
    }
    if use_constraints and bool(self.costs.abs().max() > 0.0):
      penalty = torch.zeros_like(adv)
      for i, beta in enumerate(penalty_factors):
        a = adv_c[..., i]
        std = a.std()
        if std > 1e-6:
          penalty += beta / (1.0 - gamma) * (a - a.mean()) / std
      adv = adv - penalty
      info["constraint_active"] = 1.0
      info["constraint_adv_std"] = float(penalty.std())
    self.advantages = adv
    return info

  def minibatches(self, num_minibatches: int, num_epochs: int, teacher_mask):
    """Yield dicts of flattened [T*N] samples. ``teacher`` marks teacher-group rows."""
    batch = self.T * self.N
    size = batch // num_minibatches
    flat = {
      "actions": self.actions.flatten(0, 1),
      "log_prob": self.log_prob.flatten(0, 1),
      "mu": self.mu.flatten(0, 1),
      "sigma": self.sigma.flatten(0, 1),
      "values": self.values.flatten(0, 1),
      "returns": self.returns.flatten(0, 1),
      "cost_returns": self.cost_returns.flatten(0, 1),
      "advantages": self.advantages.flatten(0, 1),
      "z_student": self.z_student.flatten(0, 1),
      "teacher": teacher_mask.unsqueeze(0).expand(self.T, -1).flatten(0, 1),
    }
    for group, tensor in self.obs.items():
      flat["obs_" + group] = tensor.flatten(0, 1)
    for _ in range(num_epochs):
      perm = torch.randperm(batch, device=self.device)
      for i in range(num_minibatches):
        idx = perm[i * size : (i + 1) * size]
        yield {k: v[idx] for k, v in flat.items()}
