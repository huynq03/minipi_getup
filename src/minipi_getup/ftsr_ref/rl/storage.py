"""Rollout storage and the force-guided mixed advantage (paper Eq. 5-8).

    A_bar_t = A_t - sum_i beta_i ( J_t,Ci + E[A_t,Ci] / (1 - gamma) )      (Eq. 8)

Resolved ambiguities (FTSR_REFERENCE_AUDIT_V2.md Sec. 9; choices and their hand-
computed checks in validate.py, test 21):

1. A_Ci estimator: GAE with a zero baseline (V_C = 0), i.e. the lambda-discounted
   cost sum. GAE does not require a learned baseline; a baseline only reduces
   variance. The paper names no cost critic and the release has none, so none is
   added (prompt rule 33: no extra machinery unless mathematically required).
2. J_t,Ci ("the constraint function value at time step t in the trajectory"): the
   discounted cost-to-go from t, sum_k gamma^k C_{t+k}, truncated at episode ends and
   at the rollout end (zero bootstrap, consistent with 1). Alternatives kept for the
   ambiguity tests: ``"immediate"`` (C_t) and ``"none"``.
3. Standardization: "prior to gradient calculation using the advantage functions" ->
   A and each A_Ci are standardized over the whole batch (the release standardizes A
   over the whole storage); J is not standardized.
4. Cost units: C1 = |F| / F_max, C2 = |T| / (T_max pi), the per-step mean over the
   physics steps. Each lies in [0, 1], so J <= 1 / (1 - gamma) = 100 and with
   beta = 0.001 the J term is <= 0.1 and the standardized A_C term is 0.1 per unit,
   against a unit-variance A. (In raw newtons J would reach 1e4 and beta = 0.001 would
   dominate A; this unit choice is a hypothesis and changes beta's meaning if
   altered.)
5. Teacher / student groups: one joint batch (the release standardizes jointly).

After t_tag every cost is exactly 0: A_C and J vanish and A_bar = standardized A.
"""

from __future__ import annotations

import torch


def gae(rew, val, last_val, dones, gamma: float, lam: float):
  """GAE over [T, N]; returns (returns, advantages)."""
  T = rew.shape[0]
  adv = torch.zeros_like(rew)
  last = torch.zeros_like(rew[0])
  for t in reversed(range(T)):
    next_val = last_val if t == T - 1 else val[t + 1]
    not_done = 1.0 - dones[t]
    delta = rew[t] + not_done * gamma * next_val - val[t]
    last = delta + not_done * gamma * lam * last
    adv[t] = last
  return adv + val, adv


def discounted_cost_to_go(cost, dones, gamma: float):
  """sum_k gamma^k C_{t+k} within the episode and the rollout, [T, N]."""
  out = torch.zeros_like(cost)
  run = torch.zeros_like(cost[0])
  for t in reversed(range(cost.shape[0])):
    run = cost[t] + (1.0 - dones[t]) * gamma * run
    out[t] = run
  return out


def standardize(x: torch.Tensor) -> torch.Tensor:
  std = x.std()
  if not torch.isfinite(std) or std < 1e-8:
    return torch.zeros_like(x)
  return (x - x.mean()) / (std + 1e-8)


def mixed_advantage(
  adv_r: torch.Tensor,
  costs: torch.Tensor,
  dones: torch.Tensor,
  gamma: float,
  lam: float,
  betas: tuple[float, ...],
  j_mode: str = "cost_return",
  standardize_cost_adv: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Eq. 8. adv_r: [T, N] reward GAE; costs: [T, N, m]. Returns (A_bar, info)."""
  a = standardize(adv_r)
  info: dict[str, float] = {}
  zeros = torch.zeros_like(adv_r[0])
  for i, beta in enumerate(betas):
    c = costs[..., i]
    _, a_c = gae(c, torch.zeros_like(c), zeros, dones, gamma, lam)
    if standardize_cost_adv:
      a_c = standardize(a_c)
    if j_mode == "cost_return":
      j = discounted_cost_to_go(c, dones, gamma)
    elif j_mode == "immediate":
      j = c
    elif j_mode == "none":
      j = torch.zeros_like(c)
    else:
      raise ValueError(j_mode)
    penalty = beta * (j + a_c / (1.0 - gamma))
    a = a - penalty
    info[f"J_mean_{i}"] = float(j.mean())
    info[f"penalty_std_{i}"] = float(penalty.std())
    info[f"penalty_mean_{i}"] = float(penalty.mean())
    info[f"cost_mean_{i}"] = float(c.mean())
  return a, info


class RolloutStorage:
  def __init__(
    self,
    T: int,
    N: int,
    dims: dict[str, int],
    num_actions: int,
    latent: int,
    num_costs: int,
    device,
  ):
    self.T, self.N, self.device = T, N, device

    def z(*shape):
      return torch.zeros(T, N, *shape, device=device)

    self.obs = {g: z(d) for g, d in dims.items()}
    self.z_t = z(latent)
    self.z_s = z(latent)
    self.actions = z(num_actions)
    self.log_prob = z()
    self.mu = z(num_actions)
    self.sigma = z(num_actions)
    self.rewards = z()
    self.costs = z(num_costs)
    self.dones = z()
    self.values = z()
    self.returns = z()
    self.advantages = z()
    self.step = 0

  def add(self, obs: dict[str, torch.Tensor], **items: torch.Tensor) -> None:
    t = self.step
    for g, x in obs.items():
      self.obs[g][t].copy_(x)
    for k, v in items.items():
      getattr(self, k)[t].copy_(v.view_as(getattr(self, k)[t]))
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def compute(
    self, last_value, gamma, lam, betas, use_constraints, j_mode, standardize_cost_adv
  ) -> dict[str, float]:
    self.returns, adv_r = gae(
      self.rewards, self.values, last_value, self.dones, gamma, lam
    )
    if use_constraints:
      self.advantages, info = mixed_advantage(
        adv_r, self.costs, self.dones, gamma, lam, betas, j_mode, standardize_cost_adv
      )
      info["constraint_active"] = 1.0
    else:
      self.advantages = standardize(adv_r)
      info = {"constraint_active": 0.0}
    return info

  def minibatches(self, num_mb: int, epochs: int, teacher_mask: torch.Tensor):
    """Flattened [T*N] samples; ``teacher`` is a per-sample mask (bug Q1 fixed)."""
    B = self.T * self.N
    size = B // num_mb
    flat = {
      "actions": self.actions.flatten(0, 1),
      "log_prob": self.log_prob.flatten(0, 1),
      "mu": self.mu.flatten(0, 1),
      "sigma": self.sigma.flatten(0, 1),
      "values": self.values.flatten(0, 1),
      "returns": self.returns.flatten(0, 1),
      "advantages": self.advantages.flatten(0, 1),
      "z_t": self.z_t.flatten(0, 1),
      "z_s": self.z_s.flatten(0, 1),
      "teacher": teacher_mask.unsqueeze(0).expand(self.T, -1).flatten(0, 1),
      "row": torch.arange(self.N, device=self.device)
      .unsqueeze(0)
      .expand(self.T, -1)
      .flatten(0, 1),
    }
    for g, x in self.obs.items():
      flat["obs_" + g] = x.flatten(0, 1)
    for _ in range(epochs):
      perm = torch.randperm(B, device=self.device)
      for i in range(num_mb):
        idx = perm[i * size : (i + 1) * size]
        yield {k: v[idx] for k, v in flat.items()}
