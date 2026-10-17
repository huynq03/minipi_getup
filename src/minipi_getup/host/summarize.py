"""Print selected TensorBoard scalars of a HoST-port run at regular iterations.

python -m minipi_getup.host.summarize logs/host/Pi_ground/<run> [--every 250]
"""

from __future__ import annotations

import argparse
import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = [
  ("Train/mean_reward", "rew"),
  ("RewardGroup/task", "task"),
  ("RewardGroup/regu", "regu"),
  ("RewardGroup/style", "style"),
  ("RewardGroup/target", "target"),
  ("RewardTerm/task_orientation", "orient"),
  ("RewardTerm/task_head_height", "headR"),
  ("Robot/head_height_mean", "head"),
  ("Robot/base_height_mean", "base"),
  ("Episode_end/success", "succ"),
  ("Episode_end/final_base_height", "fbase"),
  ("Episode_end/time_to_stand", "tts"),
  ("Curriculum/force_mean", "F"),
  ("Curriculum/force_pos_frac", "F>0"),
  ("Curriculum/action_rescale_mean", "s"),
  ("Policy/mean_noise_std", "std"),
  ("PPO/kl", "kl"),
  ("Loss/learning_rate", "lr"),
  ("Robot/torque_abs_mean", "|tau|"),
  ("Robot/joint_vel_abs_mean", "|qd|"),
  ("Robot/joint_vel_abs_max", "qdmax"),
]


def load(run: str) -> dict:
  ea = EventAccumulator(glob.glob(f"{run}/events.*")[0], size_guidance={"scalars": 0})
  ea.Reload()
  return {k: {e.step: e.value for e in ea.Scalars(k)} for k in ea.Tags()["scalars"]}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("--every", type=int, default=250)
  args = ap.parse_args()
  d = load(args.run)
  steps = sorted(d["Loss/learning_rate"])
  keys = [(k, s) for k, s in KEYS if k in d]
  print("it    " + " ".join(f"{s:>7}" for _, s in keys))
  for it in steps:
    if it % args.every == 0 or it == steps[-1]:
      row = []
      for k, _ in keys:
        v = d[k].get(it)
        row.append(f"{v:7.3g}" if v is not None else "      -")
      print(f"{it:<5} " + " ".join(row))


if __name__ == "__main__":
  main()
