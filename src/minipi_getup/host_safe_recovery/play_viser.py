"""Isolated safe-recovery playback with the extended action-scale range."""
from minipi_getup.host.viewer_adapter import HoSTViewerAdapter, HoSTViewerCommands
from minipi_getup.host_safe_recovery.asset import configure_asset
from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg
from minipi_getup.host_safe_recovery.env import SafeRecoveryEnv
from minipi_getup.host_safe_recovery.evaluate import make_cfg


class SafeRecoveryViewerCommands(HoSTViewerCommands):
  def create_gui(self, server, env_index, **kwargs):
    self.env_index = env_index
    self.force_control = server.gui.add_checkbox('Pull force (disabled)', False)
    self.scale_control = server.gui.add_slider(
      'Action scale', min=SafeRecoveryCfg.safe_recovery.action_scale_min,
      max=1.0, step=0.001, initial_value=self.adapter.action_scale,
    )
    self.status = server.gui.add_html('')


class SafeRecoveryViewerAdapter(HoSTViewerAdapter):
  def __init__(self, host, action_scale):
    super().__init__(host, action_scale)
    self.command_manager = SafeRecoveryViewerCommands(self)


def main():
  configure_asset()
  import minipi_getup.host.env as host_env
  import minipi_getup.host.play_viser as player
  # Process-local adaptation only; original HoST files stay unchanged.
  host_env.LeggedRobot_Pi = SafeRecoveryEnv
  player.HoSTViewerAdapter = SafeRecoveryViewerAdapter
  player.make_eval_cfg = lambda dr, pull, scale, episode: make_cfg(dr, scale, episode)
  player.main()


if __name__ == '__main__':
  main()
