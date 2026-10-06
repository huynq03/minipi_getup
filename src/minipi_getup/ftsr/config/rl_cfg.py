"""FTSR runner configuration (paper Sec. III, getup_gym ``GetUpPPO``)."""

from dataclasses import dataclass

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class FtsrRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "FtsrRunner"

  # Networks (paper Table I; latent size from getup_gym ``layer_size``).
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  encoder_hidden_dims: tuple[int, ...] = (256, 128)
  latent_dim: int = 12
  activation: str = "elu"
  init_noise_std: float = 1.0
  min_noise_std: float = 0.05

  # Teacher/student split: paper 3000 teacher + 1000 student of 4000.
  teacher_fraction: float = 0.75

  # PPO (getup_gym GetUpPPO.algorithm_config).
  learning_rate: float = 1.0e-3
  schedule: str = "adaptive"
  desired_kl: float = 0.01
  gamma: float = 0.99
  lam: float = 0.95
  clip_param: float = 0.2
  entropy_coef: float = 0.01
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  max_grad_norm: float = 1.0

  # Force-guided constraint (Eq. 8). beta = 0.001 is the paper's baseline penalty
  # factor; 0.02 is its "excessive" ablation.
  use_force_guidance: bool = True
  penalty_factors: tuple[float, float] = (0.001, 0.001)

  # Student encoder MSE (getup_gym student_encoder_alg_config).
  student_learning_rate: float = 1.0e-3
  student_num_learning_epochs: int = 5
  student_num_mini_batches: int = 4
  student_max_grad_norm: float = 0.8

  # Weights-only initialization (the r_w pretraining checkpoint). Unlike --resume,
  # the iteration counter and the assist schedule start at 0.
  init_checkpoint: str = ""


def ftsr_runner_cfg() -> FtsrRunnerCfg:
  return FtsrRunnerCfg(
    experiment_name="minipi_ftsr",
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=8000,
    save_interval=50,
    # Bounds the per-step target offset to 4 x 0.13 rad.
    clip_actions=4.0,
    seed=42,
  )


def ftsr_walk_runner_cfg() -> FtsrRunnerCfg:
  cfg = ftsr_runner_cfg()
  cfg.max_iterations = 400
  cfg.save_interval = 25
  return cfg
