"""Play a HoST checkpoint with MJLab's standard Viser controls."""
from __future__ import annotations

import argparse
import viser
from mjlab.viewer import ViserPlayViewer
from minipi_getup.host.evaluate import DEV, load_policy, make_eval_cfg
from minipi_getup.host.viewer_adapter import HoSTViewerAdapter


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', required=True)
  ap.add_argument('--num-envs', type=int, default=1)
  ap.add_argument('--dr', choices=['off', 'on'], default='off')
  ap.add_argument('--action-scale', type=float, default=0.25)
  ap.add_argument('--episode-s', type=float, default=10.0)
  ap.add_argument('--port', type=int, default=8080)
  args = ap.parse_args()
  from minipi_getup.host.env import LeggedRobot_Pi
  cfg = make_eval_cfg(args.dr == 'on', True, args.action_scale, args.episode_s)
  cfg.curriculum.force = 0.0
  host = LeggedRobot_Pi(cfg, args.num_envs, DEV)
  policy = load_policy(args.checkpoint, host)
  env = HoSTViewerAdapter(host, args.action_scale)
  env.reset()
  server = viser.ViserServer(port=args.port, label='HoST / MJLab')
  viewer = ViserPlayViewer(env, policy.act_inference, viser_server=server)
  print(f'[host-play-viser] {args.checkpoint}; open http://localhost:{args.port}', flush=True)
  try:
    viewer.run()
  finally:
    server.stop()
    env.close()


if __name__ == '__main__':
  main()
