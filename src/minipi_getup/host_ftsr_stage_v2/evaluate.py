"""Deterministic supine evaluation for v1 and v2 checkpoints under one identical setup.

Both checkpoints run in StageEnvV2 (same plant, observations, controller and episode as
v1; rewards do not affect a deterministic rollout). Strict success is v1's definition
(anatomical pose, overshoot < 0.015 rad, 1 s hold). "Stage 3 entry" uses the v2 entry
rule for both checkpoints. Force assistance is OFF; DR is off or on.

python -m minipi_getup.host_ftsr_stage_v2.evaluate --checkpoint CKPT --action-scale 0.99 \
  [--dr off|on] [--num-envs 128] [--episodes 2] [--video DIR] [--out JSON]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

STAND_HEIGHT = 0.3173  # Mini-Pi HoST stand threshold (host/env.py STAND_HEIGHT)


def build(checkpoint, action_scale, dr, num_envs):
  from warp._src.build import init_kernel_cache

  init_kernel_cache("/tmp/host-supine-narrow-warp")
  from minipi_getup.host.evaluate import load_policy

  from .config import StageCfgV2
  from .env import StageEnvV2

  c = StageCfgV2()
  c.noise.add_noise = False
  c.curriculum.pull_force = False
  c.curriculum.force = 0
  c.control.action_scale = action_scale
  if not dr:
    for k in dir(c.domain_rand):
      if k.startswith("randomize_") or k == "delay":
        setattr(c.domain_rand, k, False)
  env = StageEnvV2(c, num_envs)
  return env, load_policy(str(checkpoint), env)


@torch.no_grad()
def rollout(env, policy, episodes, record_env=None):
  logic = env.stage_logic
  hold_s = logic.cfg.hold_s
  out, trace = [], []
  for ep in range(episodes):
    obs, _ = env.reset()
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    snap = {}
    t_stand = torch.full((n,), float("nan"), device=env.device)
    for _ in range(int(env.max_episode_length) + 2):
      obs, _, _, done, _ = env.step(policy.act_inference(obs))
      alive &= ~done.bool()  # done rows are already reset: keep pre-reset snapshot
      if not alive.any():
        break
      f, last = env.features, logic.last
      t = (env.real_episode_length_buf.float() - env.unactuated_time) * env.dt
      up_now = (last["up"] > 0.8) & (env.root_states[:, 2] > STAND_HEIGHT) & (t > 0)
      t_stand = torch.where(alive & up_now & t_stand.isnan(), t, t_stand)
      cur = dict(
        legacy=(env.ever_stood & ~env.fell_after_stand).float(),
        upright=(logic.upright_hold >= hold_s).float(),
        strict=(logic.hold >= hold_s).float(),
        strict_ever=logic.ever_stable.float(),
        direct=((logic.hold >= hold_s) & ~logic.prone_seen & ~logic.side_seen).float(),
        prone=logic.prone_seen.float(),
        stage3=(logic.stage >= 2).float(),
        stage3_s=logic.time_in_stage[:, 2],
        overshoot_max=env.max_overshoot,
        final_overshoot=f["overshoot"],
        final_margin_score=last["margin_score"],
        final_pose_max=f["pose_max"],
        final_symmetry=f["symmetry_rms"],
        final_height=f["height"],
        near_limit=env.near_steps / env.policy_steps.clamp(min=1),
        peak_torque=env.peak_torque.max(-1).values,
        peak_qdot=env.peak_joint_vel.max(-1).values,
        time_to_stand=t_stand,
      )
      for k, v in cur.items():
        snap[k] = v.clone() if k not in snap else torch.where(alive, v, snap[k])
      if record_env is not None and ep == 0 and bool(alive[record_env]):
        i = record_env
        trace.append(
          dict(
            qpos=env.sim.data.qpos[i].cpu().numpy().copy(),
            t=float(t[i]),
            stage=int(logic.stage[i]),
            z=float(f["height"][i]),
            up=float(last["up"][i]),
            over=float(f["overshoot"][i]),
            margin=float(last["margin_score"][i]),
            tq=float(env.torques[i].abs().max()),
            strict=bool(logic.hold[i] >= hold_s),
          )
        )
    out.append({k: v.cpu().numpy() for k, v in snap.items()})
  cat = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
  tts = cat["time_to_stand"]
  stood = ~np.isnan(tts)
  summary = {
    "legacy_success": cat["legacy"].mean(),
    "upright_success": cat["upright"].mean(),
    "strict_stable_success": cat["strict"].mean(),
    "strict_ever_stable": cat["strict_ever"].mean(),
    "direct_recovery_force_off": cat["direct"].mean(),
    "ever_prone": cat["prone"].mean(),
    "stage3_entry_fraction": cat["stage3"].mean(),
    "stage3_seconds_mean": cat["stage3_s"].mean(),
    "joint_overshoot_max": cat["overshoot_max"].max(),
    "joint_overshoot_p95": np.quantile(cat["overshoot_max"], 0.95),
    "final_overshoot_median": np.median(cat["final_overshoot"]),
    "final_margin_score_median": np.median(cat["final_margin_score"]),
    "final_pose_max_median": np.median(cat["final_pose_max"]),
    "final_symmetry_rms_median": np.median(cat["final_symmetry"]),
    "final_height_median": np.median(cat["final_height"]),
    "near_joint_limit_fraction": cat["near_limit"].mean(),
    "peak_torque_p95": np.quantile(cat["peak_torque"], 0.95),
    "peak_joint_velocity_p95": np.quantile(cat["peak_qdot"], 0.95),
    "time_to_stand_median_s": float(np.median(tts[stood])) if stood.any() else None,
    "stood_fraction": stood.mean(),
    "samples": len(tts),
  }
  return {k: (float(v) if v is not None else None) for k, v in summary.items()}, trace


def write_video(env, trace, path, title):
  from minipi_getup.host.render import Renderer
  from minipi_getup.host.render import write_video as write

  r = Renderer(env.sim.mj_model)
  frames = [
    Renderer.overlay(
      r.frame(s["qpos"]),
      [
        f"{title}  t={s['t']:+.2f}s (actuation at 0)  stage {s['stage'] + 1}",
        f"base z {s['z']:.3f} m  up {s['up']:+.2f}  max|tau| {s['tq']:.1f} N m",
        f"overshoot {s['over']:.3f} rad  margin score {s['margin']:.2f}  strict {s['strict']}",
        "deterministic, force off",
      ],
    )
    for s in trace
  ]
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  write(str(path), frames, fps=int(round(1 / env.dt)))


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--action-scale", type=float, required=True)
  ap.add_argument("--dr", choices=("off", "on"), default="off")
  ap.add_argument("--num-envs", type=int, default=128)
  ap.add_argument("--episodes", type=int, default=2)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--video", type=Path, help="mp4 path for env 0, episode 0")
  ap.add_argument("--out", type=Path, help="json path")
  a = ap.parse_args()
  torch.manual_seed(a.seed)
  env, policy = build(a.checkpoint, a.action_scale, a.dr == "on", a.num_envs)
  summary, trace = rollout(env, policy, a.episodes, 0 if a.video else None)
  summary = dict(
    checkpoint=str(a.checkpoint),
    action_scale=a.action_scale,
    dr=a.dr,
    force="off",
    policy="deterministic",
    **summary,
  )
  if a.video:
    write_video(env, trace, a.video, a.checkpoint.stem)
    summary["video"] = str(a.video)
  print(json.dumps(summary, indent=2))
  if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
