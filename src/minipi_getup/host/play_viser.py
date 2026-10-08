"""Play a HoST checkpoint with MJLab's standard Viser controls."""
from __future__ import annotations

import argparse
import hashlib
import torch
import viser
from mjlab.viewer import ViserPlayViewer
from minipi_getup.host.evaluate import DEV, load_policy, make_eval_cfg
from minipi_getup.host.viewer_adapter import HoSTViewerAdapter


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', required=True)
  ap.add_argument('--num-envs', type=int, default=1)
  ap.add_argument('--dr', choices=['off', 'on'], default='off')
  ap.add_argument('--action-scale', type=float, help='Default: checkpoint curriculum, otherwise 0.25')
  ap.add_argument('--plant', choices=['auto', 'real', 'host'], default='auto')
  ap.add_argument('--episode-s', type=float, default=10.0)
  ap.add_argument('--port', type=int, default=8080)
  args = ap.parse_args()
  from warp._src.build import init_kernel_cache
  init_kernel_cache('/tmp/host-supine-narrow-warp')
  import minipi_getup.host.env as host_env
  saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
  info = saved.get('infos') or {}
  plant = info.get('plant', 'host') if args.plant == 'auto' else args.plant
  if 'plant' in info and plant != info['plant']:
    ap.error('Selected plant differs from the checkpoint training plant.')
  if info.get('asset_id') == 'cl_pai' or plant == 'real':
    from minipi_getup.host.clpai_asset import use_clpai_asset
    asset = use_clpai_asset()
    if info.get('asset_sha256') and hashlib.sha256(asset.read_bytes()).hexdigest() != info['asset_sha256']:
      ap.error('cl_pai XML differs from the checkpoint asset.')
  if args.action_scale is None:
    scale = info.get('curriculum_state', {}).get('action_rescale')
    args.action_scale = float(scale.mean()) if scale is not None else 0.25
  from minipi_getup.host.env import LeggedRobot_Pi
  cfg = make_eval_cfg(args.dr == 'on', True, args.action_scale, args.episode_s)
  host_env.TORQUE_LIMITS = None
  if plant == 'real':
    from minipi_getup.host.train_clpai import REAL_KP, REAL_KD, REAL_TORQUE
    cfg.control.stiffness, cfg.control.damping = dict(REAL_KP), dict(REAL_KD)
    host_env.TORQUE_LIMITS = dict(REAL_TORQUE)
  cfg.curriculum.force = 0.0
  host = LeggedRobot_Pi(cfg, args.num_envs, DEV)
  policy = load_policy(args.checkpoint, host)
  env = HoSTViewerAdapter(host, args.action_scale)
  env.reset()
  server = viser.ViserServer(port=args.port, label='HoST / MJLab')
  viewer = ViserPlayViewer(env, policy.act_inference, viser_server=server)
  print(f'[host-play-viser] {args.checkpoint}; plant={plant}; beta={args.action_scale:.4f}; pull OFF; open http://localhost:{args.port}', flush=True)
  try:
    viewer.run()
  finally:
    server.stop()
    env.close()


if __name__ == '__main__':
  main()
