"""Replay a HoST checkpoint on the narrow or asset-zoo robot, without training."""
import argparse
import csv
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Default: newest resume checkpoint')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--action-scale', type=float, help='Default: saved curriculum mean')
    parser.add_argument('--episode-s', type=float, default=10.0)
    parser.add_argument('--asset', choices=['narrow', 'cl-pai'], default='narrow')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = list(root.glob('resume_*/model*.pt'))
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else root/'logs/model_460_final.pt'
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')

    import warp as wp
    # Reuse the solver kernels already compiled successfully by this experiment.
    wp.config.kernel_cache_dir = '/tmp/host-supine-narrow-warp'
    import torch
    import minipi_getup.host.env as env
    from minipi_getup.host.play_viser import main as viewer

    scale = args.action_scale
    if scale is None:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = (saved.get('infos') or {}).get('curriculum_state', {})
        if 'action_rescale' in state:
            scale = state['action_rescale'].float().mean().item()
        elif checkpoint.resolve() == (root/'logs/model_460_final.pt').resolve():
            with (root/'training.csv').open() as f:
                scale = float(list(csv.DictReader(f))[-1]['action_scale'])
        else:
            parser.error('No saved action scale; supply --action-scale explicitly.')
        del saved
    if not 0.25 <= scale <= 1.0:
        parser.error('--action-scale must be between 0.25 and 1.0 for this viewer.')
    if args.asset == 'cl-pai':
        from minipi_getup.host.clpai_asset import use_clpai_asset
        use_clpai_asset()
    else:
        env.ASSET_XML = root/'assets/pi_12dof_host_narrow.xml'
    print(f'Checkpoint: {checkpoint.resolve()}\nAsset: {env.ASSET_XML}\nPull force OFF; DR OFF; action scale {scale:.4f}', flush=True)
    sys.argv = ['host-play-viser', '--checkpoint', str(checkpoint.resolve()),
                '--num-envs', '1', '--dr', 'off', '--action-scale', str(scale),
                '--episode-s', str(args.episode_s), '--port', str(args.port)]
    viewer()


if __name__ == '__main__':
    main()
