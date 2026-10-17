"""Train ``Mjlab-Getup-HoST-MiniPi`` (HoST ``legged_gym/scripts/train.py`` equivalent).

uv run host-train --run-name host_reproduction_v1 [--num-envs 4096] [--max-iterations N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from minipi_getup.host import TASK_ID
from minipi_getup.host.config import PiCfg, PiCfgPPO, class_to_dict

LOG_ROOT = os.path.join(os.getcwd(), "logs", "host")


def set_seed(seed: int) -> None:
  """legged_gym.utils.helpers.set_seed."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def main() -> None:
  p = argparse.ArgumentParser(description=f"Train {TASK_ID}")
  p.add_argument("--run-name", default="")
  p.add_argument("--num-envs", type=int, default=None)
  p.add_argument("--max-iterations", type=int, default=None)
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--resume", default=None, help="checkpoint to resume (model+optim)")
  p.add_argument("--log-dir", default=None, help="reuse an existing log dir")
  args = p.parse_args()

  env_cfg, train_cfg = PiCfg(), PiCfgPPO()
  num_envs = args.num_envs or env_cfg.env.num_envs
  if args.max_iterations is not None:
    train_cfg.runner.max_iterations = args.max_iterations
  train_cfg.runner.run_name = args.run_name
  seed = train_cfg.seed if args.seed is None else args.seed
  set_seed(seed)

  from minipi_getup.host.env import LeggedRobot_Pi
  from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner

  env = LeggedRobot_Pi(env_cfg, num_envs, args.device)
  log_dir = args.log_dir or os.path.join(
    LOG_ROOT,
    train_cfg.runner.experiment_name,
    datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
  )
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, "config.json"), "w") as f:
    json.dump(
      {
        "task": TASK_ID,
        "num_envs": num_envs,
        "seed": seed,
        "env_cfg": class_to_dict(env_cfg),
        "train_cfg": class_to_dict(train_cfg),
      },
      f,
      indent=2,
      default=str,
    )
  train_cfg_dict = class_to_dict(train_cfg)
  runner = OnPolicyRunner(env, env_cfg, train_cfg_dict, log_dir, device=args.device)
  num_iters = train_cfg.runner.max_iterations
  if args.resume:
    runner.load(args.resume)
    num_iters = train_cfg.runner.max_iterations - runner.current_learning_iteration
    print(f"resumed {args.resume} at iteration {runner.current_learning_iteration}")
  print(f"[{TASK_ID}] log dir {log_dir}", flush=True)
  runner.learn(
    num_learning_iterations=num_iters,
    init_at_random_ep_len=train_cfg.runner.init_at_random_ep_len,
  )


if __name__ == "__main__":
  main()
