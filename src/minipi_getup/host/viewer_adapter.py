"""Expose the direct HoST environment to MJLab's standard playback viewer."""
from types import SimpleNamespace

from mjlab.viewer import ViewerConfig


class HoSTViewerCommands:
  active_terms = ('HoST playback',)

  def __init__(self, adapter):
    self.adapter = adapter
    self.force_control = self.scale_control = self.status = None
    self.env_index = lambda: 0

  def create_gui(self, server, env_index, **kwargs):
    self.env_index = env_index
    self.force_control = server.gui.add_checkbox('Pull force (15 N, HoST gate)', False)
    self.scale_control = server.gui.add_slider(
      'Action scale', min=0.25, max=1.0, step=0.01,
      initial_value=self.adapter.action_scale,
    )
    self.status = server.gui.add_html('')

  def create_debug_vis_gui(self, server, **kwargs):
    pass

  def on_viewer_pause(self, paused):
    pass

  def apply_gui_reset(self, env_ids):
    return False

  def apply(self):
    env = self.adapter.host
    env.force.fill_(15.0 if self.force_control is not None and self.force_control.value else 0.0)
    scale = self.scale_control.value if self.scale_control is not None else self.adapter.action_scale
    env.action_rescale.fill_(float(scale))

  def update_status(self):
    if self.status is None:
      return
    env = self.adapter.host
    i = self.env_index()
    self.status.content = (
      f't={env.real_episode_length_buf[i].item()*env.dt:.2f} s<br/>'
      f'base={env.root_states[i, 2].item():.3f} m; '
      f'g_z={env.projected_gravity[i, 2].item():+.2f}<br/>'
      f'head-feet={env.old_headheight[i, 0].item():.3f} m<br/>'
      f'pull={env.force[i, 0].item():.1f} N; '
      f'action scale={env.action_rescale[i, 0].item():.3f}<br/>'
      f'peak torque={env.torques[i].abs().max().item():.1f} Nm; '
      f'peak qdot={env.dof_vel[i].abs().max().item():.1f} rad/s'
    )


class HoSTViewerRewards:
  def __init__(self, host):
    self.host = host

  def get_visualizable_terms(self):
    return []

  def get_active_iterable_terms(self, env_idx):
    return [(name, [self.host.rew_buf[env_idx, i].item()])
            for i, name in enumerate(self.host.reward_groups)]


class HoSTViewerAdapter:
  def __init__(self, host, action_scale):
    self.host = host
    self.action_scale = action_scale
    self.num_envs, self.device = host.num_envs, host.device
    self.sim, self.scene = host.sim, host.scene
    self.step_dt = host.dt
    self.cfg = SimpleNamespace(viewer=ViewerConfig(distance=1.5, azimuth=120, elevation=15))
    self.command_manager = HoSTViewerCommands(self)
    self.reward_manager = HoSTViewerRewards(host)

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    return self.host.get_observations()

  def step(self, actions):
    self.command_manager.apply()
    result = self.host.step(actions)
    self.command_manager.update_status()
    return result

  def reset(self, env_ids=None):
    self.command_manager.apply()
    if env_ids is None:
      result = self.host.reset()
    else:
      self.host.reset_idx(env_ids)
      self.host.compute_observations()
      result = self.host.get_observations(), self.host.get_privileged_observations()
    self.command_manager.update_status()
    return result

  def close(self):
    pass  # HoST's Simulation has no close API; viewer owns its server/threads.
