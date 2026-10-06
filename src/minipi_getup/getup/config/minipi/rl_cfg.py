"""RL configuration for HighTorque Mini-Pi getup task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def minipi_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for HighTorque Mini-Pi getup task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="minipi_getup",
    wandb_project="minipi_getup",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3_000,
  )


def minipi_getup_safe_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config for the hardware-safe get-up: raw actions clipped to +-0.8.

  With the 0.25 relative action scale that bounds each step's PD target offset to
  0.2 rad: at most ~12 Nm from the P term (kp <= 60) and ~10 rad/s of target motion.
  The same clip must be applied on the robot. (+-0.6 was too tight to sit up.)
  """
  cfg = minipi_getup_ppo_runner_cfg()
  cfg.clip_actions = 0.8
  return cfg
