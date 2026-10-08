"""Recovery analysis of one checkpoint (simulation, evaluation tooling only).

Step 1, one process per mode (same env seed -> identical initial conditions):

  python -m minipi_getup.ftsr_ref.analyze_recovery run --checkpoint CKPT \
      --mode {det,stoch,assist} --out DIR [--per-pose 128] [--seed 2150] [--tc 0.2]

  det    deterministic student (mean action), no assistance
  stoch  student mean + N(0, std) action noise (the policy's own std), no assistance
  assist deterministic student, Eq. 4 frozen at ``tc``

Step 2: ``report --out DIR`` writes plots, videos and summary.md from the runs.

Same rollout as evaluate.py (play cfg, poses by env index, 2 s passive + 18 s policy).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import minipi_getup  # noqa: F401
from minipi_getup.ftsr_ref.config.env_cfg import STAGE_HEIGHTS
from minipi_getup.ftsr_ref.config.robot import MOTOR_HYPOTHESES
from minipi_getup.ftsr_ref.evaluate import (
  DEV,
  STAND_UP,
  SUPPORT_GROUPS,
  Recorder,
  SupportTracker,
  _attach,
  build_env,
)
from minipi_getup.ftsr_ref.export import load_model
from minipi_getup.ftsr_ref.mdp.resets import FALLEN_POSE_NAMES, POSE_KEY
from minipi_getup.ftsr_ref.rl.modules import StudentPolicy

TASK = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless"  # v2 semantics
H1, H2 = STAGE_HEIGHTS[0], STAGE_HEIGHTS[1]
H_SUCCESS = 0.9 * STAGE_HEIGHTS[2]  # evaluate.py success height (0.31 m)
HOLD_S = 1.0  # recovery time: first 1 s held above h1 and upright
LAST_S = 3.0


def run(args) -> None:
  model = load_model(args.checkpoint).to(DEV)
  npose = len(FALLEN_POSE_NAMES)
  n = args.per_pose * npose

  def mutate(cfg):
    cfg.events["reset_pose"].params["by_env_index"] = True
    cfg.seed = args.seed

  torch.manual_seed(args.seed)
  tc = args.tc if args.mode == "assist" else None
  env = build_env(TASK, n, "H-conservative", mutate, tc)
  t = env.action_manager.get_term("joint_pos")
  pose = env.extras[POSE_KEY].clone()
  m = MOTOR_HYPOTHESES["H-conservative"]
  rec = Recorder(env, pose, npose, m.tau_rated, m.tau_cap)
  _attach(env, rec)
  support = SupportTracker(env)
  policy = StudentPolicy(model).to(DEV).eval()
  std = model.std.detach().clamp(min=1e-4)
  gen = torch.Generator(device=DEV).manual_seed(args.seed + 1)
  robot = env.scene["robot"]
  T = int(env.max_episode_length) - 1
  h = torch.zeros(n, T, device=DEV)
  up = torch.zeros(n, T, device=DEV)
  sup = torch.zeros(n, T, len(SUPPORT_GROUPS), dtype=torch.bool, device=DEV)
  fz = torch.zeros(n, T, device=DEV)
  qpos = torch.zeros(n, T, env.sim.data.qpos.shape[1], device=DEV)
  stage_log = []
  st = env.extras["ftsr_stage"]
  t.monitor.clear()
  obs = env.get_observations()
  with torch.no_grad():
    for k in range(T):
      a = policy(obs["policy"])
      if args.mode == "stoch":
        a = a + std * torch.randn(a.shape, generator=gen, device=DEV)
      obs, *_ = env.step(a)
      d = robot.data
      h[:, k] = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
      up[:, k] = -d.projected_gravity_b[:, 2]
      sup[:, k] = support.touching()
      f = getattr(t, "force", None) if t.cfg.assist is not None else None
      if f is not None:
        fz[:, k] = f[:, 2]
      qpos[:, k] = env.sim.data.qpos
      stage_log.append((st.stage, st.h_cmd))
  phys = {
    name: rec.group_summary(g, pose == g) for g, name in enumerate(FALLEN_POSE_NAMES)
  }
  per_env = {
    "qd_max": rec.ep_qd_max,
    "limit_overshoot": (-rec.margin_min).clamp(min=0.0),
    "track_err_max": rec.err_max,
  }
  os.makedirs(args.out, exist_ok=True)
  np.savez_compressed(
    os.path.join(args.out, f"{args.mode}.npz"),
    pose=pose.cpu().numpy(),
    h=h.cpu().numpy(),
    up=up.cpu().numpy(),
    support=sup.cpu().numpy(),
    fz=fz.cpu().numpy(),
    qpos=qpos.cpu().numpy().astype(np.float32),
    origins=env.scene.env_origins.cpu().numpy(),
    passive=t.cfg.passive_steps,
    stage=np.array(stage_log),
    dt=env.step_dt,
    **{k: v.cpu().numpy() for k, v in per_env.items()},
  )
  meta = {
    "checkpoint": os.path.abspath(args.checkpoint),
    "mode": args.mode,
    "tc": tc,
    "seed": args.seed,
    "per_pose": args.per_pose,
    "policy_std": std.cpu().tolist(),
    "slew_saturation_fraction": t.monitor.summary(t.joint_names).get(
      "slew_saturation_fraction", 0.0
    ),
    "physics": phys,
  }
  with open(os.path.join(args.out, f"{args.mode}_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
  env.close()
  print(f"[analyze] {args.mode} done", flush=True)


# ---------------------------------------------------------------------------
# Analysis


def first_hold(ok: np.ndarray, w: int) -> np.ndarray:
  """Index of the first run of ``w`` consecutive True per row, -1 if none."""
  n, T = ok.shape
  c = np.concatenate([np.zeros((n, 1)), np.cumsum(ok, 1)], 1)
  run = (c[:, w:] - c[:, :-w]) == w  # run starting at i
  return np.where(run.any(1), run.argmax(1), -1)


def first_cross(h: np.ndarray, thr: float) -> np.ndarray:
  above = h > thr
  return np.where(above.any(1), above.argmax(1), -1)


def analyze(z) -> dict:
  p, dt = int(z["passive"]), float(z["dt"])
  h, up, sup = z["h"][:, p:], z["up"][:, p:], z["support"][:, p:]
  last = int(round(LAST_S / dt))
  hold = int(round(HOLD_S / dt))
  hf, uf = h[:, -last:].mean(1), up[:, -last:].mean(1)
  feet = sup[..., 0]
  other = sup[..., 2:].any(-1)
  shin = sup[..., 1]
  feet_only = feet & ~shin & ~other
  recovered = (hf > H1) & (uf > STAND_UP)
  strict = ((h[:, -last:] > H_SUCCESS) & (up[:, -last:] > STAND_UP)).all(1)
  ok = (h > H1) & (up > STAND_UP)
  t_rec = first_hold(ok, hold)
  hmax = h.max(1)
  fell_after = (t_rec >= 0) & ~recovered
  final_feet_only = feet_only[:, -last:].mean(1) > 0.9
  cat = np.full(len(h), "recovered", dtype=object)
  nr = ~recovered
  cat[nr & (hmax < 0.15)] = "never_lift (max h < 0.15 m)"
  cat[nr & (hmax >= 0.15) & (hmax < H1)] = "partial_lift (0.15 m <= max h < h1)"
  cat[nr & (hmax >= H1) & (hf < H1) & ~fell_after] = "reached_h1_not_held"
  cat[nr & fell_after] = "fell_after_recovery (held 1 s, then down)"
  cat[nr & (hf >= H1) & (uf <= STAND_UP)] = (
    "high_but_tilted (final > h1, tilt > 18 deg)"
  )
  # Physical state at the first crossing of h1 / h2.
  cross = {}
  for name, thr in (("h1", H1), ("h2", H2), ("h_success", H_SUCCESS)):
    i = first_cross(h, thr)
    hit = i >= 0
    ii = np.clip(i, 0, h.shape[1] - 1)
    rows = np.arange(len(h))
    cross[name] = {
      "reached": hit,
      "feet_only": feet_only[rows, ii] & hit,
      "shin": shin[rows, ii] & hit,
      "torso_or_hip": (sup[rows, ii, 3] | sup[rows, ii, 4]) & hit,
      "up": np.where(hit, up[rows, ii], np.nan),
    }
  return {
    "recovered": recovered,
    "strict": strict,
    "t_rec": np.where(t_rec >= 0, t_rec * dt, np.nan),
    "hmax": hmax,
    "hf": hf,
    "uf": uf,
    "frac_h1": (h > H1).mean(1),
    "frac_h2": (h > H2).mean(1),
    "frac_hs": (h > H_SUCCESS).mean(1),
    "final_feet_only": final_feet_only,
    "cat": cat,
    "cross": cross,
    "h": h,
    "dt": dt,
  }


def pct(x) -> str:
  return f"{100.0 * float(np.mean(x)):.1f}%"


def report(args) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  out = args.out
  modes = [m for m in ("det", "stoch", "assist") if os.path.exists(f"{out}/{m}.npz")]
  Z = {m: np.load(f"{out}/{m}.npz", allow_pickle=True) for m in modes}
  M = {m: json.load(open(f"{out}/{m}_meta.json")) for m in modes}
  A = {m: analyze(Z[m]) for m in modes}
  label = {
    "det": "deterministic, no assist",
    "stoch": "stochastic (policy std), no assist",
    "assist": f"deterministic, assist tc={M.get('assist', {}).get('tc')}",
  }
  same_init = all(
    np.allclose(Z[m]["qpos"][:, 0], Z[modes[0]]["qpos"][:, 0], atol=1e-5) for m in modes
  )
  os.makedirs(f"{out}/plots", exist_ok=True)
  L: list[str] = []
  w = L.append
  ck = M[modes[0]]["checkpoint"]
  w("# model_2150 recovery evaluation (simulation)\n")
  w(f"- checkpoint: `{ck}`")
  w(
    f"- task `{TASK}` (v2 stage semantics), H-conservative plant, student policy; "
    f"{M[modes[0]]['per_pose']} envs per pose, env seed {M[modes[0]]['seed']}; "
    f"identical initial states across modes: {same_init}"
  )
  w(
    "- recovered = last 3 s mean base height > h1 (0.19 m) and mean tilt < 18 deg; "
    "strict success = evaluate.py criterion (base > 0.31 m and tilt < 18 deg over the "
    "whole last 3 s); recovery time = first 1 s held above h1 and upright, from the "
    "end of the 2 s passive window"
  )
  w(f"- policy std per joint: {np.round(M[modes[0]]['policy_std'], 2).tolist()}\n")

  w("## Recovery per pose\n")
  w(
    "| mode | pose | recovered | strict 0.31 m | time to recover median [p10, p90] s | "
    "time >h1 | >h2 | >0.31 | max h p50 | final upright cos | feet-only at end |"
  )
  w("|---|---|---|---|---|---|---|---|---|---|---|")
  for m in modes:
    a, pz = A[m], Z[m]["pose"]
    for g, name in enumerate(FALLEN_POSE_NAMES):
      r = pz == g
      tr = a["t_rec"][r & a["recovered"]]
      tt = (
        f"{np.nanmedian(tr):.2f} [{np.nanpercentile(tr, 10):.2f}, "
        f"{np.nanpercentile(tr, 90):.2f}]"
        if tr.size
        else "-"
      )
      w(
        f"| {m} | {name} | {pct(a['recovered'][r])} | {pct(a['strict'][r])} | {tt} | "
        f"{pct(a['frac_h1'][r])} | {pct(a['frac_h2'][r])} | {pct(a['frac_hs'][r])} | "
        f"{np.median(a['hmax'][r]):.3f} | {a['uf'][r].mean():.2f} | "
        f"{pct(a['final_feet_only'][r & a['recovered']]) if (r & a['recovered']).any() else '-'} |"
      )
  w("")

  w("## Failure categories (share of all episodes of the pose)\n")
  cats = [
    "never_lift (max h < 0.15 m)",
    "partial_lift (0.15 m <= max h < h1)",
    "reached_h1_not_held",
    "fell_after_recovery (held 1 s, then down)",
    "high_but_tilted (final > h1, tilt > 18 deg)",
    "recovered",
  ]
  w("| mode | pose | " + " | ".join(cats) + " |")
  w("|---|---|" + "---|" * len(cats))
  for m in modes:
    for g, name in enumerate(FALLEN_POSE_NAMES):
      r = Z[m]["pose"] == g
      w(
        f"| {m} | {name} | " + " | ".join(pct(A[m]["cat"][r] == c) for c in cats) + " |"
      )
  w("")

  w("## Joint speed, torque, joint limits (actuated time, physics-step resolution)\n")
  w(
    "| mode | pose | qd max | qd p95 (worst joint) | qd p99 (worst joint) | "
    "qd>3 | qd>4 | tau max | near cap (>=15.2 Nm) | >6 Nm | limit overshoot max | "
    "episodes overshoot >0.05 rad | slew sat |"
  )
  w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
  for m in modes:
    for g, name in enumerate(FALLEN_POSE_NAMES):
      s = M[m]["physics"][name]
      r = Z[m]["pose"] == g
      ov = Z[m]["limit_overshoot"][r]
      w(
        f"| {m} | {name} | {max(s['qd_max'].values()):.1f} | "
        f"{max(s['qd_p95'].values()):.2f} | {max(s['qd_p99'].values()):.2f} | "
        f"{100 * s['qd_frac_above_3']:.1f}% | {100 * s['qd_frac_above_4']:.2f}% | "
        f"{max(s['tau_max'].values()):.1f} | {100 * s['tau_frac_near_cap']:.2f}% | "
        f"{100 * s['tau_frac_above_rated']:.1f}% | {ov.max():.3f} | "
        f"{pct(ov > 0.05)} | {M[m]['slew_saturation_fraction']:.2f} |"
      )
  w("")

  w("## Do h1 = 0.19 m and h2 = 0.276 m mark physical stages?\n")
  w(
    "State at the first crossing of each height, and P(recovered | height reached), "
    "pooled over poses.\n"
  )
  w(
    "| mode | height | reached | feet only at crossing | shins touching | "
    "torso/hip touching | upright cos at crossing (median) | P(recovered given reached) |"
  )
  w("|---|---|---|---|---|---|---|---|")
  for m in modes:
    a = A[m]
    for hn, thr in (("h1", H1), ("h2", H2), ("h_success", H_SUCCESS)):
      c = a["cross"][hn]
      reach = c["reached"]
      if reach.any():
        w(
          f"| {m} | {hn} {thr:.3f} m | {pct(reach)} | {pct(c['feet_only'][reach])} | "
          f"{pct(c['shin'][reach])} | {pct(c['torso_or_hip'][reach])} | "
          f"{np.nanmedian(c['up'][reach]):.2f} | {pct(a['recovered'][reach])} |"
        )
      else:
        w(f"| {m} | {hn} {thr:.3f} m | 0.0% | - | - | - | - | - |")
  w(
    "\nStatic geometry (cl_pai.xml): upright kneeling on the shins reaches 0.197 m, "
    "the deepest flat-foot squat with upright torso 0.188 m, nominal stance 0.345 m.\n"
  )

  # Plots: height trajectories per pose and mode.
  fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
  colors = {"det": "C0", "stoch": "C1", "assist": "C2"}
  for g, name in enumerate(FALLEN_POSE_NAMES):
    ax = axes[g]
    for m in modes:
      hh = A[m]["h"][Z[m]["pose"] == g]
      tt = np.arange(hh.shape[1]) * A[m]["dt"]
      ax.plot(tt, np.median(hh, 0), color=colors[m], label=label[m])
      ax.fill_between(
        tt,
        np.percentile(hh, 10, 0),
        np.percentile(hh, 90, 0),
        color=colors[m],
        alpha=0.15,
      )
    for y, s in ((H1, "h1"), (H2, "h2"), (H_SUCCESS, "0.31")):
      ax.axhline(y, color="k", lw=0.6, ls="--")
      ax.text(17.9, y + 0.004, s, ha="right", fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("time after passive window [s]")
  axes[0].set_ylabel("base height [m] (median, p10-p90)")
  axes[0].legend(fontsize=8, loc="upper left")
  fig.tight_layout()
  fig.savefig(f"{out}/plots/height_trajectories.png", dpi=110)
  plt.close(fig)

  fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), sharey=True)
  axes = np.atleast_1d(axes)
  for ax, m in zip(axes, modes, strict=True):
    bottom = np.zeros(4)
    for c in cats:
      v = np.array([np.mean(A[m]["cat"][Z[m]["pose"] == g] == c) for g in range(4)])
      ax.bar(FALLEN_POSE_NAMES, v, bottom=bottom, label=c)
      bottom += v
    ax.set_title(label[m])
    ax.set_ylim(0, 1)
  axes[0].set_ylabel("share of episodes")
  axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
  fig.tight_layout()
  fig.savefig(f"{out}/plots/outcome_categories.png", dpi=110)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(6, 4))
  for m in modes:
    ax.hist(A[m]["hmax"], bins=60, range=(0.05, 0.35), histtype="step", label=label[m])
  for y in (H1, H2, H_SUCCESS):
    ax.axvline(y, color="k", lw=0.6, ls="--")
  ax.set_xlabel("max base height in episode [m]")
  ax.set_ylabel("episodes")
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(f"{out}/plots/max_height_hist.png", dpi=110)
  plt.close(fig)
  w("## Plots\n")
  w("![heights](plots/height_trajectories.png)\n")
  w("![outcomes](plots/outcome_categories.png)\n")
  w("![max height](plots/max_height_hist.png)\n")

  # Videos: per pose, deterministic no-assist: first recovered and first failed env
  # (lowest env index), plus the most common failure category's first example.
  if not args.no_video and "det" in modes:
    from minipi_getup.ftsr_ref.visual_demo import Renderer, render_env

    z, a = Z["det"], A["det"]
    mj_model = _scene_model()
    rend = Renderer(mj_model)
    r = {
      "qpos": z["qpos"],
      "origins": z["origins"],
      "h": z["h"],
      "fz": z["fz"],
      "passive": int(z["passive"]),
      "dt": float(z["dt"]),
      "stage": [(int(s), float(c)) for s, c in z["stage"]],
    }
    o = {
      "recovered": a["recovered"],
      "strict": a["strict"],
      "feet_only": a["final_feet_only"],
      "h_final": a["hf"],
      "up_final": a["uf"],
    }
    vdir = f"{out}/videos"
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(f"{out}/images", exist_ok=True)
    w("## Videos (deterministic, no assist; examples, not rates)\n")
    for g, name in enumerate(FALLEN_POSE_NAMES):
      rows = np.nonzero(z["pose"] == g)[0]
      picks = []
      win = rows[a["recovered"][rows]]
      if len(win):
        picks.append(("success", int(win[0])))
      lose = rows[~a["recovered"][rows]]
      if len(lose):
        cats_l = a["cat"][lose]
        top = max(set(cats_l), key=list(cats_l).count)
        picks.append(("failure", int(lose[cats_l == top][0])))
      for tag, i in picks:
        header = [
          "Mini-Pi FTSR v2 (simulation)",
          "checkpoint: model_2150",
          "student policy (deterministic)",
          f"pose: {name}   [{tag} example: {a['cat'][i]}]",
          "assist: none (0 N)",
          f"seed: {M['det']['seed']}  env index: {i}",
        ]
        base = f"{name}_{tag}_env{i}"
        render_env(
          rend,
          r,
          o,
          i,
          header,
          f"{vdir}/{base}.mp4",
          f"{out}/images/{base}",
        )
        w(f"- `videos/{base}.mp4`: {tag}, {a['cat'][i]}")
    w("")

  with open(f"{out}/summary.md", "w") as f:
    f.write("\n".join(L) + "\n")
  print("\n".join(L))


def _scene_model():
  """Host-side MuJoCo model of the scene (one robot + ground) for rendering."""
  env = build_env(TASK, 4, "H-conservative")
  m = env.sim.mj_model
  env.close()
  return m


def main() -> None:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("run")
  r.add_argument("--checkpoint", required=True)
  r.add_argument("--mode", choices=("det", "stoch", "assist"), required=True)
  r.add_argument("--out", required=True)
  r.add_argument("--per-pose", type=int, default=128)
  r.add_argument("--seed", type=int, default=2150)
  r.add_argument("--tc", type=float, default=0.2)
  p = sub.add_parser("report")
  p.add_argument("--out", required=True)
  p.add_argument("--no-video", action="store_true")
  args = ap.parse_args()
  run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
  main()
