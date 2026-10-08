"""FTSR networks (paper Table I, release ``modules/``).

| Model           | Input                     | Hidden          | Output |
| Actor           | z (12), o_t (48)           | [512, 256, 128] | a (12) |
| Critic          | z_t (12, teacher), s (50)  | [512, 256, 128] | V      |
| Teacher encoder | x_t (20)                   | [256, 128]      | z^t    |
| Student encoder | o_{t-4..t} (240)           | [256, 128]      | z^s    |

ELU, linear outputs, state-independent action std initialized to 1.0 (release).
Critic: the release's ``[E_t(x_t), privileged]`` for every row (teacher latent even for
student rows); the paper's Table I / Fig. 2 differ (recorded conflict, release kept).
No cost critic: the constraint advantages use a zero baseline (``storage.py``).
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  dim = in_dim
  for h in hidden:
    layers += [nn.Linear(dim, h), nn.ELU()]
    dim = h
  layers.append(nn.Linear(dim, out_dim))
  return nn.Sequential(*layers)


class FtsrModel(nn.Module):
  def __init__(
    self,
    dims: dict[str, int],
    num_actions: int,
    latent_dim: int = 12,
    actor_hidden: tuple[int, ...] = (512, 256, 128),
    critic_hidden: tuple[int, ...] = (512, 256, 128),
    encoder_hidden: tuple[int, ...] = (256, 128),
    init_noise_std: float = 1.0,
  ):
    super().__init__()
    self.dims = dict(dims)
    self.latent_dim = latent_dim
    self.teacher_encoder = mlp(dims["teacher"], encoder_hidden, latent_dim)
    self.student_encoder = mlp(dims["policy"], encoder_hidden, latent_dim)
    self.actor = mlp(latent_dim + dims["actor"], actor_hidden, num_actions)
    self.critic = mlp(latent_dim + dims["critic"], critic_hidden, 1)
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

  def actor_mean(self, z: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
    return self.actor(torch.cat((z, o_t), dim=-1))

  def value(self, z_t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return self.critic(torch.cat((z_t, s), dim=-1)).squeeze(-1)

  def ppo_parameters(self) -> list[nn.Parameter]:
    """theta (actor, std), phi (critic) and theta^t (teacher encoder), one optimizer
    and one gradient clip, as the release's TSPPO."""
    return [
      *self.actor.parameters(),
      self.std,
      *self.critic.parameters(),
      *self.teacher_encoder.parameters(),
    ]


class StudentPolicy(nn.Module):
  """Deployable policy: obs = o_{t-4..t} (frame-major) -> raw action.

  a = actor(E^s(obs), o_t) with o_t = the newest frame, i.e. obs[:, -48:].
  """

  def __init__(self, model: FtsrModel):
    super().__init__()
    self.student_encoder = model.student_encoder
    self.actor = model.actor
    self.frame_dim = model.dims["actor"]

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    z = self.student_encoder(obs)
    o_t = obs[:, -self.frame_dim :]
    return self.actor(torch.cat((z, o_t), dim=-1))
