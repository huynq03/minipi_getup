"""Train host_ftsr_stage_v1 from scratch; --resume accepts only this experiment."""

import argparse
import hashlib
import json
import subprocess
import types
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--max-iterations", type=int, default=12000)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--resume", type=Path)
  ap.add_argument(
    "--output-root", type=Path, default=Path("results/host/host_ftsr_stage_v1")
  )
  args = ap.parse_args()
  if min(args.num_envs, args.max_iterations) < 1:
    ap.error("Counts must be positive")
  if (
    subprocess.check_output(
      ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    )
    .decode()
    .strip()
  ):
    ap.error("GPU has compute processes. Close viewers/other training before starting.")
  from warp._src.build import init_kernel_cache

  init_kernel_cache("/tmp/host-supine-narrow-warp")
  import torch

  from minipi_getup.host.config import class_to_dict
  from minipi_getup.host.train import set_seed
  from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner

  from .config import StageCfg, StagePPOCfg, StageSettings
  from .env import StageEnv
  from .geometry import XML, calibrate

  torch.set_num_threads(4)
  set_seed(args.seed)
  cfg = StageCfg()
  pcfg = StagePPOCfg()
  pcfg.seed = args.seed
  pcfg.runner.max_iterations = args.max_iterations
  digest = hashlib.sha256(XML.read_bytes()).hexdigest()
  contract = dict(
    experiment="host_ftsr_stage_v1",
    plant="real",
    asset_id="cl_pai",
    asset_sha256=digest,
    stage_settings=asdict(StageSettings()),
    env_cfg=class_to_dict(cfg),
    train_cfg=class_to_dict(pcfg),
  )
  # Compare the reward/plant contract before constructing a resumed env.
  if args.resume:
    saved = torch.load(args.resume, map_location="cpu", weights_only=False)
    prior = saved["infos"]
    for k in ("experiment", "plant", "asset_sha256", "stage_settings", "env_cfg"):
      if prior.get(k) != contract[k]:
        ap.error(f"Resume contract mismatch: {k}")
    if (
      saved.get("it") if saved.get("it") is not None else saved["iter"]
    ) >= args.max_iterations:
      ap.error("No iterations remaining")
  env = StageEnv(cfg, args.num_envs)
  logdir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  logdir.mkdir(parents=True, exist_ok=False)
  (logdir / "config.json").write_text(
    json.dumps(
      dict(contract, num_envs=args.num_envs, seed=args.seed, geometry=calibrate()),
      indent=2,
    )
  )
  runner = OnPolicyRunner(env, cfg, class_to_dict(pcfg), str(logdir), device="cuda:0")
  if args.resume:
    state = runner.load(str(args.resume))["curriculum_state"]
    for name in ("force", "action_rescale"):
      value = state[name].to(env.device)
      dest = getattr(env, name)
      dest.copy_(value if value.shape == dest.shape else value.mean().expand_as(dest))
    env.compute_observations()
  save = runner.save
  log = runner.log
  completed = [runner.current_learning_iteration]

  def persist(self, path, infos=None, it=None):
    info = dict(
      contract,
      curriculum_state={
        k: getattr(env, k).detach().cpu() for k in ("force", "action_rescale")
      },
    )
    # Physics restarts on resume; per-episode stages/high-water/hold must reset
    # together with it. They are deliberately NOT restored onto a new supine pose.
    actual = completed[0]
    save(path, infos=info, it=actual)

  def log_step(self, locs):
    completed[0] = locs["it"] + 1
    log(locs)

  runner.save = types.MethodType(persist, runner)
  runner.log = types.MethodType(log_step, runner)
  print(f"host_ftsr_stage_v1: {logdir}; scratch={not bool(args.resume)}", flush=True)
  try:
    runner.learn(
      args.max_iterations - runner.current_learning_iteration,
      init_at_random_ep_len=pcfg.runner.init_at_random_ep_len,
    )
  except KeyboardInterrupt:
    runner.save(str(logdir / "model_interrupted.pt"), it=completed[0])
    print(
      "Saved interrupted resume checkpoint; may contain a partial update.", flush=True
    )
  finally:
    if runner.writer:
      runner.writer.flush()
      runner.writer.close()


if __name__ == "__main__":
  main()
