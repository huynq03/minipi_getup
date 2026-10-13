"""Measure recovery routes and reward contributions before first standing.

Uses the same reset seeds as measure_recovery_drag.py. A prone visit means
projected gravity x >= 0.8 for at least 0.1 s while not yet standing. This is
an orientation measurement, not a claim about belly contact or causality.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from minipi_getup.host.env import STAND_HEIGHT
from minipi_getup.host_safe_recovery.env import RESET_CLASSES
from minipi_getup.host_staged_recovery.evaluate import build, load_actor


@torch.inference_mode()
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--action-scale", type=float)
  parser.add_argument("--seed", type=int, default=20261013)
  parser.add_argument("--num-envs", type=int, default=100)
  parser.add_argument("--trials", type=int, default=3)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  state = saved["infos"]["curriculum_state"]
  scale = args.action_scale if args.action_scale is not None else float(state["action_rescale"].mean())
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  env, name, sha, ppo, _ = build("staged", scale, 10.0, args.num_envs, args.device)
  # Restore the success-dependent reward strength, but keep force/scale fixed.
  env.penalty_success_ema.copy_(state["penalty_success_ema"].to(env.device))
  # env.test does not freeze these inherited reset callbacks. Freeze explicitly.
  env.update_force_curriculum = lambda env_ids: None
  env._penalty_ema_ids = lambda env_ids: env_ids[:0]
  actor = load_actor(args.checkpoint, env, name, sha, ppo, args.device)
  n, dev = env.num_envs, env.device
  records, examples = [], {}
  hold_steps = round(0.1 / env.dt)
  for trial in range(args.trials):
    torch.manual_seed(args.seed + trial)
    np.random.seed(args.seed + trial)
    obs, _ = env.reset()
    classes = env.reset_class.clone()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    stood = torch.zeros_like(alive)
    prone_run = torch.zeros(n, device=dev)
    supine_run = torch.zeros_like(prone_run)
    prone_max = torch.zeros_like(prone_run)
    supine_max = torch.zeros_like(prone_run)
    first_g = torch.full((n, 3), float("nan"), device=dev)
    t_stand = torch.full((n,), float("nan"), device=dev)
    t_prone = torch.full_like(t_stand, float("nan"))
    travel = torch.zeros(n, 2, device=dev)
    weight = env.sim.model.body_mass[:, env.robot.indexing.body_ids.long()].sum(1) * 9.81
    sums = {}
    histories = {i: [] for i in range(n) if int(classes[i]) in (2, 3)}
    for step in range(int(10.0 / env.dt)):
      obs, *_ = env.step(actor.act_inference(obs))
      alive &= ~env.reset_buf.bool()
      active = alive & ~stood & (env.real_episode_length_buf > env.unactuated_time)
      g = env.projected_gravity
      t = (env.real_episode_length_buf - env.unactuated_time) * env.dt
      first = active & ~torch.isfinite(first_g[:, 0])
      first_g[first] = g[first]
      fl, fr, _ = env._ground_support_forces()
      loaded = (torch.stack((fl, fr), 1) >= 0.05 * weight[:, None]) & (env.sole_clearance < 0.005)
      travel += env.foot_speed_xy * loaded * env.dt * active[:, None]
      prone_run = torch.where(active & (g[:, 0] >= 0.8), prone_run + 1, 0)
      supine_run = torch.where(active & (g[:, 0] <= -0.8), supine_run + 1, 0)
      prone_max = torch.maximum(prone_max, prone_run)
      supine_max = torch.maximum(supine_max, supine_run)
      hit_prone = (prone_run >= hold_steps) & ~torch.isfinite(t_prone)
      t_prone[hit_prone] = t[hit_prone] - (hold_steps - 1) * env.dt
      values = dict(env.term_values, task_product=env.rew_buf[:, 0])
      for key, value in values.items():
        value = value.reshape(n)
        if key not in sums:
          sums[key] = torch.zeros(n, device=dev)
        sums[key] += value * active
      if step % 5 == 0:
        sample = torch.cat((t[:, None], g, env.root_states[:, 2:3],
                            env.support_gate[:, None], env.ready[:, None]), 1).cpu().tolist()
        mask = active.cpu().tolist()
        for i in histories:
          if mask[i]:
            histories[i].append(sample[i])
      standing = (env.root_states[:, 2] > STAND_HEIGHT) & (-g[:, 2] > 0.8)
      hit = active & standing
      t_stand[hit] = t[hit]
      stood |= hit
      if (stood | ~alive).all():
        break
    for i in range(n):
      orientation = RESET_CLASSES[int(classes[i])]
      row = {
        "orientation": orientation,
        "stood": bool(stood[i]),
        "time_to_stand_s": float(t_stand[i]) if stood[i] else None,
        "prone_visit": bool(prone_max[i] >= hold_steps),
        "supine_visit": bool(supine_max[i] >= hold_steps),
        "first_prone_s": float(t_prone[i]) if torch.isfinite(t_prone[i]) else None,
        "max_prone_duration_s": float(prone_max[i] * env.dt),
        "drag_left_m": float(travel[i, 0]),
        "drag_right_m": float(travel[i, 1]),
        "gravity_at_actuation": first_g[i].cpu().tolist(),
        "reward_sums_before_standing": {k: float(v[i]) for k, v in sums.items()},
      }
      records.append(row)
      if orientation in ("left", "right"):
        example_key = f"{orientation}_{'prone' if row['prone_visit'] else 'other'}"
        if example_key not in examples:
          examples[example_key] = {"episode": len(records) - 1, "trajectory": histories[i]}
    print(f"Trial {trial + 1}/{args.trials}: stood {int(stood.sum())}/{n}", flush=True)
  summary = {}
  for orientation in RESET_CLASSES:
    rows = [r for r in records if r["orientation"] == orientation]
    if not rows:
      continue
    ok = [r for r in rows if r["stood"]]
    summary[orientation] = {
      "episodes": len(rows),
      "stood_fraction": mean(r["stood"] for r in rows),
      "prone_visit_fraction": mean(r["prone_visit"] for r in rows),
      "supine_visit_fraction": mean(r["supine_visit"] for r in rows),
      "already_prone_at_actuation_fraction": mean(r["gravity_at_actuation"][0] >= 0.8 for r in rows),
      "mean_time_to_stand_s": mean(r["time_to_stand_s"] for r in ok) if ok else None,
      "mean_drag_left_cm": 100 * mean(r["drag_left_m"] for r in rows),
      "mean_drag_right_cm": 100 * mean(r["drag_right_m"] for r in rows),
      "mean_reward_sums_before_standing": {
        k: mean(r["reward_sums_before_standing"][k] for r in rows) for k in sums
      },
    }
  output = {
    "checkpoint": str(args.checkpoint), "action_scale": scale, "seed": args.seed,
    "penalty_success_ema": float(state["penalty_success_ema"]),
    "prone_definition": "projected_gravity.x >= 0.8 for >= 0.1 s before first standing",
    "conditions": "No pull force, DR, additive observation noise, stance starts or velocity/drag termination; action-scale observation jitter inherited from env.py remains active.",
    "trajectory_columns": ["time_s", "gravity_x", "gravity_y", "gravity_z", "base_height_m", "support_gate", "ready"],
    "summary": summary, "examples": examples, "episodes": records,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(output, indent=2, allow_nan=False))
  print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'mean_reward_sums_before_standing'}
                    for k, v in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
  main()
