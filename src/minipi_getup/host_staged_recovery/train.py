"""Train the staged-recovery policy from scratch (fresh actor, four critics, optimizer),
or resume a checkpoint of the same variant and asset with --resume."""

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
from minipi_getup.host_safe_recovery.optimization import install_action_std_guard
from minipi_getup.host_staged_recovery import EXPERIMENT_NAME
from minipi_getup.host_staged_recovery.asset import (
  ASSET_XML,
  asset_manifest,
  configure_asset,
)
from minipi_getup.host_staged_recovery.config import (
  StagedRecoveryCfg,
  StagedRecoveryPPOCfg,
)
from minipi_getup.host_staged_recovery.env import StagedRecoveryEnv

PACKAGES = ("src/minipi_getup/host_staged_recovery", "src/minipi_getup/host_safe_recovery")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--max-iterations", type=int, default=12000)
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--min-action-std", type=float, default=0.05)
  p.add_argument("--max-action-std", type=float, default=1.0)
  p.add_argument("--allow-busy-gpu", action="store_true")
  p.add_argument("--allow-dirty", action="store_true", help="smoke tests only")
  p.add_argument("--run-name", default="")
  p.add_argument("--resume", type=Path, help="checkpoint to continue; --max-iterations are added to its iteration")
  args = p.parse_args()
  if args.num_envs < 1 or args.max_iterations < 1:
    p.error("--num-envs and --max-iterations must be positive")
  if not 0.0 < args.min_action_std <= args.max_action_std:
    p.error("action std bounds must satisfy 0 < minimum <= maximum")
  dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *PACKAGES], text=True).strip()
  if dirty and not args.allow_dirty:
    p.error(f"commit the experiment code before a real run:\n{dirty}")
  if not args.allow_busy_gpu and args.device.startswith("cuda"):
    busy = subprocess.check_output(
      ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
      p.error(f"GPU has active compute processes: {busy}")

  set_seed(args.seed)
  configure_asset()
  manifest = asset_manifest()  # hashed once: the XML may change on disk later
  cfg, pcfg = StagedRecoveryCfg(), StagedRecoveryPPOCfg()
  cfg.asset.file = str(ASSET_XML)
  pcfg.seed = args.seed
  pcfg.runner.max_iterations = args.max_iterations
  pcfg.runner.save_interval = 100
  pcfg.runner.run_name = EXPERIMENT_NAME

  env = StagedRecoveryEnv(cfg, args.num_envs, args.device)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  logdir = Path.cwd() / "logs" / EXPERIMENT_NAME / (stamp + (f"_{args.run_name}" if args.run_name else ""))
  logdir.mkdir(parents=True)
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  metadata = {
    "variant": EXPERIMENT_NAME,
    "source_commit": commit,
    "source_dirty": bool(dirty),
    "source_diff": subprocess.check_output(["git", "diff", "--", *PACKAGES], text=True),
    "reward_doc": "src/minipi_getup/host_staged_recovery/REWARDS.md",
    "asset": str(ASSET_XML),
    "asset_manifest_sha256": manifest,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "max_iterations": args.max_iterations,
    "resume": str(args.resume) if args.resume else None,
    "action_std_bounds": [args.min_action_std, args.max_action_std],
    "reward_scales_raw": class_to_dict(cfg.constraints.scales),
    "env_cfg": class_to_dict(cfg),
    "train_cfg": class_to_dict(pcfg),
  }
  (logdir / "config.json").write_text(json.dumps(metadata, indent=2, default=str))
  runner = OnPolicyRunner(env, cfg, class_to_dict(pcfg), str(logdir), device=args.device)
  install_action_std_guard(runner.alg, args.min_action_std, args.max_action_std)
  if args.resume:
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    info = checkpoint.get("infos") or {}
    if info.get("variant") != EXPERIMENT_NAME or info.get("asset_sha256") != manifest[ASSET_XML.name]:
      p.error("checkpoint is not compatible with this variant and asset")
    runner.load(str(args.resume))
    state = info.get("curriculum_state") or {}
    for name in ("force", "action_rescale", "penalty_success_ema"):
      if name in state:
        saved = state[name].to(env.device)
        getattr(env, name).copy_(saved if saved.shape == getattr(env, name).shape else saved.mean().expand_as(getattr(env, name)))

  old_save = runner.save
  def save(self, path, infos=None, it=None):
    payload = dict(infos or {})
    payload.update(
      variant=EXPERIMENT_NAME,
      asset_sha256=manifest[ASSET_XML.name],
      source_commit=commit,
      seed=args.seed,
      action_std_bounds=[args.min_action_std, args.max_action_std],
      curriculum_state={n: getattr(env, n).detach().cpu() for n in ("force", "action_rescale", "penalty_success_ema")},
    )
    old_save(path, infos=payload, it=it)
  runner.save = types.MethodType(save, runner)

  print(
    f"[{EXPERIMENT_NAME}] {'resume ' + str(args.resume) if args.resume else 'scratch'}; seed={args.seed}; commit={commit}; asset={ASSET_XML}; "
    f"sha256={manifest[ASSET_XML.name]}; logs={logdir}", flush=True
  )
  runner.learn(args.max_iterations, init_at_random_ep_len=pcfg.runner.init_at_random_ep_len)


if __name__ == "__main__":
  main()
