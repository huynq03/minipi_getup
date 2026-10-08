"""Interactive viser playback of a HoST-port checkpoint.

uv run host-play-viser --checkpoint logs/host/Pi_ground/<run>/model_N.pt \
  [--num-envs 1] [--dr off] [--port 8080]

Opens http://localhost:<port>. The policy is deterministic and runs in real time at
50 Hz. The GUI has pause, reset, pull force on/off, action scale and env selection;
defaults follow the release's eval (pull force off, noise off, action scale 0.25).
"""

from __future__ import annotations

import argparse
import time

import torch
import viser
from mjlab.viewer.viser.scene import MjlabViserScene

from minipi_getup.host.evaluate import DEV, load_policy, make_eval_cfg


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--dr", choices=["off", "on"], default="off")
  ap.add_argument("--action-scale", type=float, default=0.25)
  ap.add_argument("--episode-s", type=float, default=10.0)
  ap.add_argument("--port", type=int, default=8080)
  args = ap.parse_args()

  from minipi_getup.host.env import LeggedRobot_Pi

  cfg = make_eval_cfg(args.dr == "on", True, args.action_scale, args.episode_s)
  cfg.curriculum.force = 0.0  # pull force starts off; the GUI sets it
  env = LeggedRobot_Pi(cfg, args.num_envs, DEV)
  policy = load_policy(args.checkpoint, env)

  server = viser.ViserServer(port=args.port)
  scene = MjlabViserScene(
    server=server,
    mj_model=env.sim.mj_model,
    num_envs=env.num_envs,
    sim_model=env.sim.model,
    expanded_fields=env.sim.expanded_fields,
  )
  scene.env_idx = 0
  scene.create_scene_gui(camera_distance=1.5, camera_azimuth=120, camera_elevation=15)

  state = {"paused": False, "reset": True}
  with server.gui.add_folder("HoST play"):
    status = server.gui.add_html("")
    pause = server.gui.add_button("Pause")
    reset = server.gui.add_button("Reset (all envs)")
    force_on = server.gui.add_checkbox("Pull force (15 N, HoST gate)", False)
    scale = server.gui.add_slider(
      "Action scale", min=0.25, max=1.0, step=0.01, initial_value=args.action_scale
    )
    env_sel = server.gui.add_slider(
      "Env", min=0, max=max(env.num_envs - 1, 0), step=1, initial_value=0
    )

  @pause.on_click
  def _(_) -> None:
    state["paused"] = not state["paused"]
    pause.name = "Play" if state["paused"] else "Pause"

  @reset.on_click
  def _(_) -> None:
    state["reset"] = True

  @env_sel.on_update
  def _(_) -> None:
    scene.env_idx = int(env_sel.value)
    scene.request_update()

  print(f"[host-play-viser] open http://localhost:{args.port}", flush=True)
  obs = env.get_observations()
  with torch.inference_mode():
    while True:
      t0 = time.time()
      if state["reset"]:
        env.reset_idx(torch.arange(env.num_envs, device=DEV))
        obs, *_ = env.step(torch.zeros(env.num_envs, env.num_actions, device=DEV))
        state["reset"] = False
      if not state["paused"]:
        env.force[:] = 15.0 if force_on.value else 0.0
        env.action_rescale[:] = float(scale.value)
        obs, *_ = env.step(policy.act_inference(obs))
        scene.update(env.sim.data, int(env_sel.value))
      i = int(env_sel.value)
      t = env.real_episode_length_buf[i].item() * env.dt
      status.content = (
        f"<b>{args.checkpoint.split('/')[-1]}</b> env {i}<br>"
        f"t = {t:.2f} s (actuated from 0.62 s)<br>"
        f"base z = {env.root_states[i, 2].item():.3f} m, "
        f"g_z = {env.projected_gravity[i, 2].item():+.2f}<br>"
        f"head-feet = {env.old_headheight[i, 0].item():.3f} m<br>"
        f"pull force = {env.pending_force[i, 0, 2].item():.1f} N, "
        f"action scale = {env.action_rescale[i, 0].item():.2f}<br>"
        f"max |torque| = {env.torques[i].abs().max().item():.1f} N m, "
        f"max |qd| = {env.dof_vel[i].abs().max().item():.1f} rad/s"
      )
      time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == "__main__":
  main()
