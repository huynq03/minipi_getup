"""Runner configuration: getup_gym ``GetUpPPO`` (FTSR_REFERENCE_AUDIT_V2.md Sec. 2)."""

from dataclasses import dataclass

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class FtsrRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "FtsrRunner"

  # Networks (release; latent = layer_size 12).
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  encoder_hidden_dims: tuple[int, ...] = (256, 128)
  latent_dim: int = 12
  init_noise_std: float = 1.0

  # Teacher / student split: 3000 of 4000 (rows [0, 3000) teacher).
  teacher_fraction: float = 0.75

  # PPO.
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

  # Force-guided objective (Eq. 5-8, storage.py). beta = 0.001: paper baseline.
  use_force_guidance: bool = True
  penalty_factors: tuple[float, float] = (0.001, 0.001)
  cost_j_mode: str = "cost_return"
  standardize_cost_advantage: bool = True

  # Student encoder MSE (release student_encoder_alg_config).
  student_learning_rate: float = 1.0e-3
  student_num_learning_epochs: int = 5
  student_num_mini_batches: int = 4
  student_max_grad_norm: float = 0.8

  # Weights-only initialization (walking checkpoint).
  init_checkpoint: str = ""

  # Periodic no-assist evaluation (separate process), every eval_every iterations.
  eval_every: int = 0
  eval_at: tuple[int, ...] = ()
  """Explicit evaluation iterations; after the last one, every ``eval_every``."""
  eval_task: str = ""
  eval_envs_per_pose: int = 64


def recovery_runner_cfg() -> FtsrRunnerCfg:
  return FtsrRunnerCfg(
    experiment_name="minipi_ftsr_ref",
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=8000,
    save_interval=50,
    clip_actions=None,  # raw clip lives in the action term (deploy order)
    seed=42,
    eval_every=500,
    eval_at=(500, 1000, 1500, 2000, 2500, 2750, 3000, 3500),
    eval_task="Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless",
  )


def walk_runner_cfg() -> FtsrRunnerCfg:
  cfg = recovery_runner_cfg()
  cfg.experiment_name = "minipi_ftsr_ref_walk"
  cfg.max_iterations = 400
  cfg.save_interval = 25
  cfg.eval_every = 0
  cfg.eval_at = ()
  cfg.eval_task = ""
  return cfg
