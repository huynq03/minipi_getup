"""FTSR networks (paper Table I).

| Model           | Input       | Hidden          | Output |
| Actor pi        | z_t, o_t    | [512, 256, 128] | a_t    |
| Critic phi      | z_t, s_t    | [512, 256, 128] | V_t    |
| Teacher enc E^t | x_t         | [256, 128]      | z^t_t  |
| Student enc E^s | o_{t:t-H}   | [256, 128]      | z^s_t  |

The cost critic estimates the value of the two constraint costs (assist force and
torque) for the advantages A_Ci in Eq. 8. Paper and reference code don't specify
its architecture, so it copies the critic. Latent size 12, as in getup_gym's
``layer_size``.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str = "elu"):
  act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
  layers: list[nn.Module] = []
  dim = in_dim
  for h in hidden:
    layers += [nn.Linear(dim, h), act()]
    dim = h
  layers.append(nn.Linear(dim, out_dim))
  return nn.Sequential(*layers)


class FtsrModel(nn.Module):
  def __init__(
    self,
    actor_obs_dim: int,
    student_obs_dim: int,
    teacher_obs_dim: int,
    critic_obs_dim: int,
    num_actions: int,
    latent_dim: int = 12,
    actor_hidden: tuple[int, ...] = (512, 256, 128),
    critic_hidden: tuple[int, ...] = (512, 256, 128),
    encoder_hidden: tuple[int, ...] = (256, 128),
    num_costs: int = 2,
    activation: str = "elu",
    init_noise_std: float = 1.0,
  ):
    super().__init__()
    self.latent_dim = latent_dim
    self.teacher_encoder = mlp(teacher_obs_dim, encoder_hidden, latent_dim, activation)
    self.student_encoder = mlp(student_obs_dim, encoder_hidden, latent_dim, activation)
    self.actor = mlp(latent_dim + actor_obs_dim, actor_hidden, num_actions, activation)
    self.critic = mlp(latent_dim + critic_obs_dim, critic_hidden, 1, activation)
    self.cost_critic = mlp(
      latent_dim + critic_obs_dim, critic_hidden, num_costs, activation
    )
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

  def actor_mean(self, z: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
    return self.actor(torch.cat((z, actor_obs), dim=-1))

  def value(self, z: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.critic(torch.cat((z, critic_obs), dim=-1))

  def cost_value(self, z: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.cost_critic(torch.cat((z, critic_obs), dim=-1))

  def ppo_parameters(self) -> list[nn.Parameter]:
    """Updated by the PPO loss: theta, phi, E^t (paper Eq. 6) and the cost critic."""
    return [
      *self.actor.parameters(),
      self.std,
      *self.critic.parameters(),
      *self.cost_critic.parameters(),
      *self.teacher_encoder.parameters(),
    ]


class StudentPolicy(nn.Module):
  """Deployable policy: a = pi(E^s(o_{t:t-H}), o_t), proprioception only."""

  def __init__(self, model: FtsrModel):
    super().__init__()
    self.student_encoder = model.student_encoder
    self.actor = model.actor

  def forward(self, student_obs: torch.Tensor, actor_obs: torch.Tensor):
    z = self.student_encoder(student_obs)
    return self.actor(torch.cat((z, actor_obs), dim=-1))
