"""Train the isolated HoST-only safe-recovery policy from scratch or its own checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from datetime import datetime
from pathlib import Path

import torch

from minipi_getup.host.config import class_to_dict
from minipi_getup.host.train import set_seed
from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner
from minipi_getup.host_safe_recovery import EXPERIMENT_NAME
from minipi_getup.host_safe_recovery.asset import (
  ASSET_XML,
  asset_sha256,
  configure_asset,
)
from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg, SafeRecoveryPPOCfg
from minipi_getup.host_safe_recovery.env import SafeRecoveryEnv
from minipi_getup.host_safe_recovery.optimization import install_action_std_guard


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--max-iterations", type=int, default=12000)
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--resume", type=Path)
  p.add_argument("--min-action-std", type=float, default=0.05)
  p.add_argument("--max-action-std", type=float, default=1.0)
  p.add_argument("--allow-busy-gpu", action="store_true")
  args = p.parse_args()
  if args.num_envs < 1 or args.max_iterations < 1:
    p.error("--num-envs and --max-iterations must be positive")
  if not 0.0 < args.min_action_std <= args.max_action_std:
    p.error("action std bounds must satisfy 0 < minimum <= maximum")
  if not args.allow_busy_gpu and args.device.startswith("cuda"):
    busy = subprocess.check_output(
      ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
      p.error(f"GPU has active compute processes: {busy}")

  set_seed(args.seed)
  configure_asset()
  cfg, pcfg = SafeRecoveryCfg(), SafeRecoveryPPOCfg()
  cfg.asset.file = str(ASSET_XML)
  pcfg.seed = args.seed
  pcfg.runner.max_iterations = args.max_iterations
  pcfg.runner.save_interval = 100
  pcfg.runner.run_name = EXPERIMENT_NAME

  env = SafeRecoveryEnv(cfg, args.num_envs, args.device)
  logdir = Path.cwd() / "logs" / EXPERIMENT_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
  logdir.mkdir(parents=True)
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  metadata = {
    "variant": EXPERIMENT_NAME,
    "source_commit": commit,
    "asset": str(ASSET_XML),
    "asset_sha256": asset_sha256(),
    "seed": args.seed,
    "num_envs": args.num_envs,
    "max_iterations": args.max_iterations,
    "resume": str(args.resume) if args.resume else None,
    "action_std_bounds": [args.min_action_std, args.max_action_std],
    "env_cfg": class_to_dict(cfg),
    "train_cfg": class_to_dict(pcfg),
  }
  (logdir / "config.json").write_text(json.dumps(metadata, indent=2, default=str))
  runner = OnPolicyRunner(env, cfg, class_to_dict(pcfg), str(logdir), device=args.device)
  install_action_std_guard(runner.alg, args.min_action_std, args.max_action_std)
  if args.resume:
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    info = checkpoint.get("infos") or {}
    if info.get("variant") != EXPERIMENT_NAME or info.get("asset_sha256") != asset_sha256():
      p.error("checkpoint is not compatible with this variant and asset")
    runner.load(str(args.resume))
    state = info.get("curriculum_state") or {}
    for name in ("force", "action_rescale"):
      if name in state:
        saved = state[name].to(env.device)
        getattr(env, name).copy_(saved if saved.shape == getattr(env, name).shape else saved.mean().expand_as(getattr(env, name)))

  old_save = runner.save
  def save(self, path, infos=None, it=None):
    payload = dict(infos or {})
    payload.update(
      variant=EXPERIMENT_NAME,
      asset_sha256=asset_sha256(),
      source_commit=commit,
      action_std_bounds=[args.min_action_std, args.max_action_std],
      curriculum_state={n: getattr(env, n).detach().cpu() for n in ("force", "action_rescale")},
    )
    old_save(path, infos=payload, it=it)
  runner.save = types.MethodType(save, runner)

  remaining = args.max_iterations - runner.current_learning_iteration
  if remaining <= 0:
    p.error("--max-iterations must exceed the checkpoint iteration")
  print(
    f"[{EXPERIMENT_NAME}] {'resume' if args.resume else 'scratch'}; "
    f"asset={ASSET_XML}; sha256={asset_sha256()}; logs={logdir}", flush=True
  )
  runner.learn(remaining, init_at_random_ep_len=pcfg.runner.init_at_random_ep_len)


if __name__ == "__main__":
  main()
