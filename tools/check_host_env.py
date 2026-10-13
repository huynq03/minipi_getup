"""Check Rev13 dependencies and step two CUDA environments without training.

Run after: source tools/activate_host_env.sh
Usage: python tools/check_host_env.py
No policy, optimizer, checkpoint, or training runner is created.
"""

import importlib.metadata
import sys

import torch
import warp as wp
from minipi_getup.host_staged_recovery.asset import asset_manifest, configure_asset
from minipi_getup.host_staged_recovery.config import StagedRecoveryCfg
from minipi_getup.host_staged_recovery.env import StagedRecoveryEnv


def main():
  print(f"Python: {sys.version.split()[0]}", flush=True)
  for package in (
    "torch",
    "mjlab",
    "mujoco",
    "mujoco-warp",
    "warp-lang",
    "tensorboard",
  ):
    print(f"{package}: {importlib.metadata.version(package)}", flush=True)
  if not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA unavailable; run this check in a terminal with GPU access."
    )
  print(
    f"GPU: {torch.cuda.get_device_name(0)}; torch CUDA: {torch.version.cuda}",
    flush=True,
  )
  wp.init()
  if not wp.get_cuda_devices():
    raise RuntimeError("Warp cannot access CUDA.")
  configure_asset()
  manifest = asset_manifest()
  print(f"Asset manifest verified: {len(manifest)} files", flush=True)
  cfg = StagedRecoveryCfg()
  assert cfg.staged.revision == "13"
  assert cfg.staged.drag_termination is False
  assert cfg.staged.action_scale_floor == 0.25
  with torch.no_grad():
    env = StagedRecoveryEnv(cfg, num_envs=2, device="cuda:0")
    env.reset()
    for _ in range(3):
      obs, privileged_obs, rewards, _, _ = env.step(
        torch.zeros((2, env.num_actions), device=env.device)
      )
      for tensor in (obs, privileged_obs, rewards):
        if tensor is not None and not torch.isfinite(tensor).all():
          raise RuntimeError("Non-finite simulation output.")
    torch.cuda.synchronize()
  print(
    "READY: Rev13 CUDA reset + 3 simulation steps passed; no training executed.",
    flush=True,
  )


if __name__ == "__main__":
  main()
