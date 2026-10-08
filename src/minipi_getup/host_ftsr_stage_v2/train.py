"""Train host_ftsr_stage_v2.

Modes (exactly one of --init-from / --resume, or neither for scratch):

- ``--init-from <v1 checkpoint>``: warm start, NOT a PPO resume. Loads the v1 actor
  (incl. action std) and the task/regu/style critics; the target critic is
  re-initialized because the target reward changed (v1 target return was ~0); the
  Adam state starts fresh (its moments belong to the old critic). The iteration
  counter and force/action-scale curriculum continue from the v1 checkpoint.
- ``--resume <v2 checkpoint>``: full resume (model, optimizer, curriculum) of v2.

python -m minipi_getup.host_ftsr_stage_v2.train --init-from \
  results/host/host_ftsr_stage_v1/<run>/model_1000.pt --max-iterations 1400
"""

import argparse
import hashlib
import json
import subprocess
import types
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

EXPERIMENT = "host_ftsr_stage_v2"
TARGET_CRITIC = 3  # reward_groups = [task, regu, style, target]


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument(
    "--max-iterations", type=int, required=True, help="total, incl. prior"
  )
  ap.add_argument("--seed", type=int, default=1)
  src = ap.add_mutually_exclusive_group()
  src.add_argument("--init-from", type=Path, help="v1 checkpoint (warm start)")
  src.add_argument("--resume", type=Path, help="v2 checkpoint (full resume)")
  ap.add_argument(
    "--output-root", type=Path, default=Path(f"results/host/{EXPERIMENT}")
  )
  ap.add_argument(
    "--allow-shared-gpu", action="store_true", help="start even if GPU has other jobs"
  )
  args = ap.parse_args()
  busy = (
    subprocess.check_output(
      ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    )
    .decode()
    .strip()
  )
  if busy and not args.allow_shared_gpu:
    ap.error(f"GPU has compute processes ({busy}); pass --allow-shared-gpu to share.")
  from warp._src.build import init_kernel_cache

  init_kernel_cache("/tmp/host-supine-narrow-warp")
  import torch

  from minipi_getup.host.config import class_to_dict
  from minipi_getup.host.train import set_seed
  from minipi_getup.host_ftsr_stage.geometry import XML, calibrate
  from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner

  from .config import StageCfgV2, StagePPOCfgV2, StageSettingsV2
  from .env import StageEnvV2

  torch.set_num_threads(4)
  set_seed(args.seed)
  cfg, pcfg = StageCfgV2(), StagePPOCfgV2()
  pcfg.seed = args.seed
  pcfg.runner.max_iterations = args.max_iterations
  digest = hashlib.sha256(XML.read_bytes()).hexdigest()
  contract = dict(
    experiment=EXPERIMENT,
    plant="real",
    asset_id="cl_pai",
    asset_sha256=digest,
    stage_settings=asdict(StageSettingsV2()),
    env_cfg=class_to_dict(cfg),
    train_cfg=class_to_dict(pcfg),
  )
  prior = None
  if args.init_from or args.resume:
    path = args.init_from or args.resume
    prior = torch.load(path, map_location="cpu", weights_only=False)
    info = prior["infos"]
    want = "host_ftsr_stage_v1" if args.init_from else EXPERIMENT
    if info.get("experiment") != want:
      ap.error(f"{path} is not a {want} checkpoint")
    for k in ("plant", "asset_sha256"):
      if info.get(k) != contract[k]:
        ap.error(f"Checkpoint {k} differs from v2")
    if args.resume:
      for k in ("stage_settings", "env_cfg"):
        if info.get(k) != contract[k]:
          ap.error(f"Resume contract mismatch: {k}")
    if int(prior["it"]) >= args.max_iterations:
      ap.error("No iterations remaining")
  env = StageEnvV2(cfg, args.num_envs)
  logdir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  logdir.mkdir(parents=True, exist_ok=False)
  mode = "init_from_v1" if args.init_from else "resume" if args.resume else "scratch"
  runner = OnPolicyRunner(env, cfg, class_to_dict(pcfg), str(logdir), device="cuda:0")
  ac = runner.alg.actor_critic
  if args.resume:
    runner.load(str(args.resume))
  elif args.init_from:
    state = prior["model_state_dict"]
    fresh = {
      k: v
      for k, v in ac.state_dict().items()
      if k.startswith(f"critics.{TARGET_CRITIC}.")
    }
    state = {**state, **fresh}
    ac.load_state_dict(state)
    runner.current_learning_iteration = int(prior["it"])
    # Optimizer: fresh Adam (constructed by the runner); not loaded on purpose. The
    # KL-adaptive learning rate is schedule state, not optimizer state: keep v1's.
    if prior.get("learning_rate") is not None:
      runner.alg.learning_rate = float(prior["learning_rate"])
      for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
  if prior is not None:
    curriculum = prior["infos"]["curriculum_state"]
    for name in ("force", "action_rescale"):
      value = curriculum[name].to(env.device)
      dest = getattr(env, name)
      dest.copy_(value if value.shape == dest.shape else value.mean().expand_as(dest))
    env.compute_observations()
  (logdir / "config.json").write_text(
    json.dumps(
      dict(
        contract,
        mode=mode,
        source_checkpoint=str(args.init_from or args.resume or ""),
        start_iteration=runner.current_learning_iteration,
        warm_start=(
          "actor + std + task/regu/style critics loaded; target critic re-initialized; "
          "fresh optimizer; curriculum restored"
          if args.init_from
          else None
        ),
        num_envs=args.num_envs,
        seed=args.seed,
        geometry=calibrate(),
      ),
      indent=2,
      default=str,
    )
  )
  save, log = runner.save, runner.log
  completed = [runner.current_learning_iteration]

  def persist(self, path, infos=None, it=None):
    info = dict(
      contract,
      mode=mode,
      curriculum_state={
        k: getattr(env, k).detach().cpu() for k in ("force", "action_rescale")
      },
    )
    save(path, infos=info, it=completed[0])

  def log_step(self, locs):
    completed[0] = locs["it"] + 1
    log(locs)

  runner.save = types.MethodType(persist, runner)
  runner.log = types.MethodType(log_step, runner)
  print(
    f"{EXPERIMENT}: {logdir}; mode={mode}; start_it={runner.current_learning_iteration}",
    flush=True,
  )
  try:
    runner.learn(
      args.max_iterations - runner.current_learning_iteration,
      init_at_random_ep_len=pcfg.runner.init_at_random_ep_len,
    )
  except KeyboardInterrupt:
    runner.save(str(logdir / "model_interrupted.pt"), it=completed[0])
    print("Saved interrupted checkpoint; may contain a partial update.", flush=True)
  finally:
    if runner.writer:
      runner.writer.flush()
      runner.writer.close()


if __name__ == "__main__":
  main()
