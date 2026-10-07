"""Print a training run's latest TensorBoard scalars in the terminal, refreshing.

Usage:
  python -m minipi_getup.ftsr.tbwatch [--run REGEX] [--every SECONDS] [--filter TEXT ...]

``--run`` picks the newest run directory under logs/rsl_rl/minipi_ftsr whose name
matches (default: the newest run). Rows are grouped by tag prefix (Episode_Reward,
Episode_Metrics, Train, ...). Reward terms are per-second averages over episodes
(see CLAUDE.md, "Reward logging").
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _run_dir(pattern: str) -> str:
  runs = sorted(glob.glob("logs/rsl_rl/minipi_ftsr/*"))
  runs = [r for r in runs if os.path.isdir(r) and re.search(pattern, r)]
  if not runs:
    raise SystemExit(f"no run matches {pattern!r}")
  return runs[-1]


def _show(run: str, filters: list[str]) -> None:
  ea = EventAccumulator(run, size_guidance={"scalars": 0})
  ea.Reload()
  tags = sorted(ea.Tags()["scalars"])
  if filters:
    tags = [t for t in tags if any(f.lower() in t.lower() for f in filters)]
  step = max((ea.Scalars(t)[-1].step for t in tags), default=0)
  print(f"\033[2J\033[H{os.path.basename(run)}  iteration {step}")
  group = None
  for tag in tags:
    prefix, _, name = tag.rpartition("/")
    if prefix != group:
      group = prefix
      print(f"\n[{group}]")
    print(f"  {name:<32} {ea.Scalars(tag)[-1].value:>12.4g}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default=".", help="regex on the run directory name")
  ap.add_argument("--every", type=float, default=30.0, help="refresh period (s)")
  ap.add_argument("--filter", nargs="*", default=[], help="only tags containing")
  args = ap.parse_args()
  run = _run_dir(args.run)
  while True:
    _show(run, args.filter)
    time.sleep(args.every)


if __name__ == "__main__":
  main()
