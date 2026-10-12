"""Staged-recovery playback in the HoST viser viewer (pull force disabled)."""
from minipi_getup.host_safe_recovery.play_viser import SafeRecoveryViewerAdapter
from minipi_getup.host_staged_recovery.asset import configure_asset
from minipi_getup.host_staged_recovery.config import StagedRecoveryCfg
from minipi_getup.host_staged_recovery.env import StagedRecoveryEnv


def make_cfg(dr, scale, episode):
  cfg = StagedRecoveryCfg()
  cfg.noise.add_noise = False
  cfg.env.episode_length_s = episode
  cfg.control.action_scale = scale
  cfg.curriculum.pull_force = False
  cfg.env.test = True
  if not dr:
    for key in dir(cfg.domain_rand):
      if key.startswith("randomize_") or key == "delay":
        setattr(cfg.domain_rand, key, False)
  return cfg


def main():
  configure_asset()
  import minipi_getup.host.env as host_env
  import minipi_getup.host.play_viser as player
  # Process-local adaptation only; original HoST files stay unchanged.
  host_env.LeggedRobot_Pi = StagedRecoveryEnv
  player.HoSTViewerAdapter = SafeRecoveryViewerAdapter
  player.make_eval_cfg = lambda dr, pull, scale, episode: make_cfg(dr, scale, episode)
  player.main()


if __name__ == "__main__":
  main()
