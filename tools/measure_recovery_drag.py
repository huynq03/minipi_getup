"""Measure loaded-foot travel before first standing, grouped by reset orientation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from minipi_getup.host.env import STAND_HEIGHT
from minipi_getup.host_safe_recovery.env import RESET_CLASSES
from minipi_getup.host_staged_recovery.evaluate import build, load_actor


@torch.inference_mode()
def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
  scale = float(saved["infos"]["curriculum_state"]["action_rescale"].mean())
  seed, n, trials, seconds = 20261013, 100, 3, 10.0
  torch.manual_seed(seed)
  np.random.seed(seed)
  env, name, sha, ppo, _ = build("staged", scale, seconds, n, "cuda:0")
  actor = load_actor(args.checkpoint, env, name, sha, ppo, "cuda:0")
  records = []
  for trial in range(trials):
    # Reuse the same initial resets across checkpoints despite different auto-resets.
    torch.manual_seed(seed + trial)
    np.random.seed(seed + trial)
    obs, _ = env.reset()
    cls = env.reset_class.clone()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    stood = torch.zeros_like(alive)
    travel = torch.zeros(n, 2, device=env.device)
    peak = torch.zeros(n, device=env.device)
    time_to_stand = torch.full((n,), float("nan"), device=env.device)
    weight = (
      env.sim.model.body_mass[:, env.robot.indexing.body_ids.long()].sum(1) * 9.81
    )
    for _ in range(int(seconds / env.dt)):
      obs, *_ = env.step(actor.act_inference(obs))
      active = alive & ~env.reset_buf.bool()
      actuated = env.real_episode_length_buf > env.unactuated_time
      rising = active & actuated & ~stood
      fl, fr, _ = env._ground_support_forces()
      loaded = (torch.stack((fl, fr), 1) >= 0.05 * weight[:, None]) & (
        env.sole_clearance < 0.005
      )
      speed = env.foot_speed_xy * loaded
      travel += speed * env.dt * rising[:, None]
      peak = torch.maximum(peak, speed.max(1).values * rising)
      standing = (env.root_states[:, 2] > STAND_HEIGHT) & (
        -env.projected_gravity[:, 2] > 0.8
      )
      hit = active & actuated & ~stood & standing
      time_to_stand[hit] = (
        env.real_episode_length_buf[hit] - env.unactuated_time
      ) * env.dt
      stood |= hit
      alive = active
      if not alive.any():
        break
    for i in range(n):
      records.append(
        {
          "orientation": RESET_CLASSES[int(cls[i])],
          "stood": bool(stood[i]),
          "drag_m_per_foot": float(travel[i].mean()),
          "peak_sliding_speed_m_s": float(peak[i]),
          "time_to_stand_s": float(time_to_stand[i]) if stood[i] else None,
        }
      )
    print(f"Trial {trial + 1}/{trials}: stood {int(stood.sum())}/{n}", flush=True)
  summary = {}
  for orientation in RESET_CLASSES:
    rows = [r for r in records if r["orientation"] == orientation]
    ok = [r for r in rows if r["stood"]]
    summary[orientation] = {
      "episodes": len(rows),
      "stood": len(ok),
      "stood_fraction": len(ok) / len(rows),
      "mean_drag_cm_per_foot_successful": float(
        np.mean([r["drag_m_per_foot"] * 100 for r in ok])
      )
      if ok
      else None,
      "median_drag_cm_per_foot_successful": float(
        np.median([r["drag_m_per_foot"] * 100 for r in ok])
      )
      if ok
      else None,
      "mean_peak_sliding_speed_m_s_successful": float(
        np.mean([r["peak_sliding_speed_m_s"] for r in ok])
      )
      if ok
      else None,
      "mean_time_to_stand_s": float(np.mean([r["time_to_stand_s"] for r in ok]))
      if ok
      else None,
    }
  result = {
    "checkpoint": str(args.checkpoint),
    "action_scale": scale,
    "seed": seed,
    "episode_s": seconds,
    "definition": "Per-foot average horizontal sole-centre path length while load >= 5% body weight and clearance < 5 mm, after actuation starts until first standing (base > STAND_HEIGHT and upright > 0.8). Ranking uses episodes that reached standing. No pull force, DR, observation noise or stance starts; deterministic policy.",
    "summary": summary,
    "episodes": records,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
  print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
  main()
