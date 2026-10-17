"""Roll out a HoST-port checkpoint deterministically and render videos.

uv run host-play --checkpoint logs/host/Pi_ground/<run>/model_N.pt --num-envs 1 \
  [--pull-force off] [--dr off] [--action-scale 0.25] [--episode-s 5] [--out DIR]

Defaults follow the release's eval (pull force off, noise off, action scale 0.25).
"""

from __future__ import annotations

import argparse
import json
import os

from minipi_getup.host.evaluate import rollout
from minipi_getup.host.render import Renderer, write_video


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--render-envs", type=int, default=None, help="how many to render")
  ap.add_argument("--dr", choices=["off", "on"], default="off")
  ap.add_argument("--pull-force", choices=["off", "on"], default="off")
  ap.add_argument("--action-scale", type=float, default=0.25)
  ap.add_argument("--episode-s", type=float, default=5.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=None)
  ap.add_argument("--tag", default="")
  args = ap.parse_args()
  n_render = min(args.render_envs or args.num_envs, args.num_envs)
  summary, traces, env = rollout(
    args.checkpoint,
    num_envs=args.num_envs,
    episodes=1,
    dr=args.dr == "on",
    pull_force=args.pull_force == "on",
    action_scale=args.action_scale,
    episode_s=args.episode_s,
    record_envs=tuple(range(n_render)),
    seed=args.seed,
  )
  ckpt = os.path.splitext(os.path.basename(args.checkpoint))[0]
  out = args.out or os.path.join(os.path.dirname(args.checkpoint), "videos")
  os.makedirs(out, exist_ok=True)
  rend = Renderer(env.sim.mj_model)
  tag = args.tag or (
    f"{ckpt}_force{args.pull_force}_dr{args.dr}_scale{args.action_scale:g}"
  )
  paths = []
  for i, tr in traces.items():
    frames = []
    for s in tr:
      img = rend.frame(s["qpos"])
      frames.append(
        Renderer.overlay(
          img,
          [
            f"{ckpt}  env {i}  t={s['t']:.2f}s  (actuated from 0.62 s)",
            f"base z {s['base_z']:.3f} m  head-feet {s['head']:.3f} m  g_z {s['pg_z']:+.2f}",
            f"pull force {s['force']:.1f} N  action scale {s['action_rescale']:.2f}",
            f"max |torque| {s['max_torque']:.1f} N m  deterministic policy",
          ],
        )
      )
    p = os.path.join(out, f"{tag}_env{i}.mp4")
    write_video(p, frames, fps=int(round(1 / env.dt)))
    paths.append(p)
  summary["videos"] = paths
  with open(os.path.join(out, f"{tag}_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
  print(
    json.dumps(
      {k: summary[k] for k in summary if k != "per_joint_peak_torque"}, indent=2
    )
  )


if __name__ == "__main__":
  main()
