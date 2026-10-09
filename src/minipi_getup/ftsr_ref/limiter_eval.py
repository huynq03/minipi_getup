"""Zero-shot evaluation of a PD-target rate limiter (evaluation only, simulation).

No training, no weight or optimizer change: the checkpoint's deterministic student
runs in the play cfg of the v2 stateless-stage task, ZERO assistance, fallen poses by
env index, one 20 s episode per env (2 s passive + 18 s policy), exactly the rollout of
``analyze_recovery periodic``. Two configurations per checkpoint and seed:

- baseline: the unchanged plant (``FtsrAction``: q* = clip(q_default + scale a_c,
  q_min, q_max), PD tau = clip(kp (q* - q) - kd qdot, +-16 Nm)).
- limited:  the same, with q* additionally rate limited per joint, after the physical
  clip and before the PD law::

      q*_t = q*_{t-1} + clip(clip(q_cmd_t, q_min, q_max) - q*_{t-1}, -L, +L)

  q*_{t-1} at the first actuated step of an episode (and in the passive window) is the
  measured joint position of that step, clipped to the range. Implemented by wrapping
  ``FtsrAction.process_actions`` in this process only: the raw action, the
  observations (``last_action`` is the raw-clipped network output, unchanged), the
  network, the gains, the torque cap and the ranges are untouched. Enforcement is
  verified at every policy step (``limiter_check`` in ``stats.json``).

``rollout`` records, at every 0.5 ms physics step, |qdot| (quantized to 0.01 rad/s,
reduced in-process to per-phase / per-joint histograms and > 20 rad/s events),
joint-limit penetration, |tau|, tracking error, torque saturation and the root
constraint force (contact proxy); per policy step h, uprightness, ground support, q*,
q, qpos (videos) and validity. Recovery metrics use ``analyze_recovery.analyze``
unchanged (recovered = last 3 s mean height > h1 and mean uprightness > cos 18 deg).

  python -m minipi_getup.ftsr_ref.limiter_eval rollout --checkpoint CKPT \
      --limit {0,0.3} --seed S --per-pose 128 --out RUN_DIR
  python -m minipi_getup.ftsr_ref.limiter_eval report --root OUT
  python -m minipi_getup.ftsr_ref.limiter_eval videos --root OUT
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

import minipi_getup  # noqa: F401
from minipi_getup.ftsr_ref.analyze_recovery import analyze as recovery_analyze
from minipi_getup.ftsr_ref.config.env_cfg import STAGE_HEIGHTS
from minipi_getup.ftsr_ref.config.robot import JOINT_NAMES, MINIPI_WEIGHT, TAU_CAP
from minipi_getup.ftsr_ref.evaluate import (
  DEV,
  STAND_UP,
  SUPPORT_GROUPS,
  SupportTracker,
  build_env,
)
from minipi_getup.ftsr_ref.export import load_model
from minipi_getup.ftsr_ref.mdp.actions import INVALID_KEY, rate_limit_target
from minipi_getup.ftsr_ref.mdp.resets import FALLEN_POSE_NAMES, POSE_KEY
from minipi_getup.ftsr_ref.rl.modules import StudentPolicy

TASK = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless"  # v2 stateless stages
H1, H2 = STAGE_HEIGHTS[0], STAGE_HEIGHTS[1]
SHORT = [x.replace("_joint", "") for x in JOINT_NAMES]
STABLE_S = 0.5  # phase D: 0.5 s continuously above h2 and upright (velocity audit)
GROUPS = ("A_lift", "B_h1_to_h2", "C_stabilize", "D_standing", "first_100ms")
THRESH = (4.0, 6.28, 10.0, 20.0)
Q = 100.0  # |qdot| quantization, 0.01 rad/s
NB = 10000  # histogram bins (0 .. 99.99 rad/s; larger values go to the last bin)
SPIKE = 20.0
SAT = 0.95 * TAU_CAP
VIOL = 0.05  # rad, episode-level joint-limit violation threshold
BIG_F = 10.0  # body weights, "large contact impulse" (root constraint force)


# ---------------------------------------------------------------------------
# Rollout


def _install_limiter(t, robot, lim: float) -> None:
  """Wrap ``t.process_actions`` with the per-joint target-rate limiter (for a task
  whose cfg has no limiter). Same function and initialization as the training
  implementation (``FtsrActionCfg.target_rate_limit``)."""
  orig = t.process_actions
  prev = {"q": None}

  def limited(actions):
    was = t.entered.clone()
    orig(actions)  # unchanged pipeline: raw clip, physical clip, passive handling
    q_meas = torch.clamp(robot.data.joint_pos[:, t._ids], t.q_min, t.q_max)
    entry = t.entered & ~was
    reinit = (t.passive | entry).unsqueeze(-1)
    base = q_meas if prev["q"] is None else torch.where(reinit, q_meas, prev["q"])
    t.rate_base = base
    t.q_star = rate_limit_target(t.q_star, base, lim)
    prev["q"] = t.q_star.clone()

  t.process_actions = limited


def rollout(args) -> None:
  from mjlab.tasks.registry import load_env_cfg

  t0 = time.time()
  if load_env_cfg(args.task, play=True).actions["joint_pos"].target_rate_limit > 0:
    if args.limit > 0.0:
      raise SystemExit(f"{args.task} already applies a limiter; --limit must be 0")
  model = load_model(args.checkpoint).to(DEV)
  npose = len(FALLEN_POSE_NAMES)
  n = args.per_pose * npose

  def mutate(cfg):
    cfg.events["reset_pose"].params["by_env_index"] = True
    cfg.seed = args.seed

  torch.manual_seed(args.seed)
  env = build_env(args.task, n, mutate)
  t = env.action_manager.get_term("joint_pos")
  robot = env.scene["robot"]
  data = env.sim.data
  support = SupportTracker(env)
  dec = int(env.cfg.decimation)
  T = int(env.max_episode_length) - 1 if args.steps <= 0 else args.steps
  S = T * dec
  nj = len(JOINT_NAMES)
  P = int(t.cfg.passive_steps)
  pose = env.extras[POSE_KEY].clone()
  origin_z = env.scene.env_origins[:, 2]
  qmin, qmax = t.q_min, t.q_max

  # Trace envs: the first ``trace_per_pose`` envs of each pose (full substep data).
  pose_np = pose.cpu().numpy()
  tr = np.concatenate(
    [np.nonzero(pose_np == g)[0][: args.trace_per_pose] for g in range(npose)]
  )
  tr_t = torch.tensor(tr, device=DEV)
  K = len(tr)

  qd_u16 = np.zeros((S, n, nj), np.uint16)
  pol = {
    "q_star": np.zeros((T, n, nj), np.float32),
    "q": np.zeros((T, n, nj), np.float32),
    "qd_step_max": np.zeros((T, n, nj), np.float16),
    "viol_step_max": np.zeros((T, n, nj), np.float16),
    "err_step_max": np.zeros((T, n, nj), np.float16),
    "tau_step_max": np.zeros((T, n, nj), np.float16),
    "sat_count": np.zeros((T, n), np.int16),
    "root_f_peak": np.zeros((T, n), np.float32),
    "h": np.zeros((T, n), np.float32),
    "up": np.zeros((T, n), np.float32),
    "support": np.zeros((T, n, len(SUPPORT_GROUPS)), bool),
    "invalid": np.zeros((T, n), bool),
    "nan_term": np.zeros((T, n), bool),
    "qpos": np.zeros((T, n, data.qpos.shape[1]), np.float32),
  }
  trace = {
    "q": np.zeros((S, K, nj), np.float32),
    "qd": np.zeros((S, K, nj), np.float32),
    "tau": np.zeros((S, K, nj), np.float32),
    "h": np.zeros((S, K), np.float32),
  }
  g = {
    "q": torch.zeros(dec, n, nj, device=DEV),
    "qd": torch.zeros(dec, n, nj, device=DEV),
    "tau": torch.zeros(dec, n, nj, device=DEV),
    "rf": torch.zeros(dec, n, device=DEV),
    "h": torch.zeros(dec, n, device=DEV),
  }
  state = {"j": 0}
  step = env.sim.step

  def hooked():
    j = state["j"]
    step()
    g["q"][j] = robot.data.joint_pos[:, t._ids]
    g["qd"][j] = robot.data.joint_vel[:, t._ids]
    g["tau"][j] = data.actuator_force[:, t._ctrl]
    g["rf"][j] = data.qfrc_constraint[:, 0:3].norm(dim=-1)
    g["h"][j] = data.qpos[:, 2] - origin_z
    state["j"] = j + 1

  env.sim.step = hooked
  cfg_lim = float(t.cfg.target_rate_limit)
  if cfg_lim > 0.0 and args.limit > 0.0:
    raise SystemExit(
      f"{args.task} already applies a {cfg_lim} rad/step limiter; --limit must be 0"
    )
  lim = cfg_lim if cfg_lim > 0.0 else float(args.limit)
  source = "task cfg" if cfg_lim > 0.0 else ("eval wrapper" if lim > 0.0 else None)
  check = {
    "limit_rad_per_step": lim,
    "limit_source": source,
    "entry_base_mismatch_max": 0.0,
    "max_step": 0.0,
    "clipped": 0,
    "joint_steps": 0,
    "outside_range": 0,
    "entry_max_from_measured": 0.0,
    "entry_envs": 0,
    "raw_action_mismatch_max": 0.0,
  }
  if source == "eval wrapper":
    _install_limiter(t, robot, lim)
  policy = StudentPolicy(model).to(DEV).eval()
  obs = env.get_observations()
  tm = env.termination_manager
  q0 = data.qpos.clone()
  with torch.no_grad():
    for k in range(T):
      a = policy(obs["policy"])
      was = t.entered.clone()
      q_before = torch.clamp(robot.data.joint_pos[:, t._ids], qmin, qmax)
      state["j"] = 0
      obs, *_ = env.step(a)
      assert state["j"] == dec
      if lim > 0.0:
        # Limiter verification at every policy step (either implementation).
        act = ~t.passive
        entry = t.entered & ~was
        if act.any():
          d = (t.q_star - t.rate_base).abs()[act]
          check["max_step"] = max(check["max_step"], float(d.max()))
          over = (t.q_cmd - t.rate_base).abs()[act] > lim + 1e-7
          check["clipped"] += int(over.sum())
          check["joint_steps"] += int(d.numel())
          inside = (t.q_star >= qmin - 1e-6) & (t.q_star <= qmax + 1e-6)
          check["outside_range"] += int((~inside[act]).sum())
        if entry.any():
          e = (t.q_star - q_before).abs()[entry]
          check["entry_max_from_measured"] = max(
            check["entry_max_from_measured"], float(e.max())
          )
          bm = float((t.rate_base - q_before).abs()[entry].max())
          check["entry_base_mismatch_max"] = max(check["entry_base_mismatch_max"], bm)
          check["entry_envs"] += int(entry.sum())
      # The limiter must not change the raw action seen by the observation.
      ac = torch.clamp(a, -t.cfg.raw_clip, t.cfg.raw_clip)
      act = ~t.passive
      if act.any():
        mm = float((t.raw_action - ac).abs()[act].max())
        check["raw_action_mismatch_max"] = max(check["raw_action_mismatch_max"], mm)
      q, qd, tau = g["q"], g["qd"], g["tau"]
      aqd = qd.abs()
      s = slice(k * dec, (k + 1) * dec)
      qd_u16[s] = (
        torch.clamp(torch.round(aqd * Q), max=65535).to(torch.int32).cpu().numpy()
      )
      viol = torch.clamp(torch.maximum(qmin - q, q - qmax), min=0.0)
      pol["qd_step_max"][k] = aqd.amax(0).cpu().numpy()
      pol["viol_step_max"][k] = viol.amax(0).cpu().numpy()
      pol["err_step_max"][k] = (t.q_star - q).abs().amax(0).cpu().numpy()
      atau = tau.abs()
      pol["tau_step_max"][k] = atau.amax(0).cpu().numpy()
      pol["sat_count"][k] = (atau >= SAT).sum((0, 2)).cpu().numpy()
      pol["root_f_peak"][k] = g["rf"].amax(0).cpu().numpy()
      pol["q_star"][k] = t.q_star.cpu().numpy()
      pol["q"][k] = q[-1].cpu().numpy()
      d = robot.data
      pol["h"][k] = (d.root_link_pos_w[:, 2] - origin_z).cpu().numpy()
      pol["up"][k] = (-d.projected_gravity_b[:, 2]).cpu().numpy()
      pol["support"][k] = support.touching().cpu().numpy()
      pol["invalid"][k] = env.extras[INVALID_KEY].cpu().numpy()
      pol["nan_term"][k] = tm.get_term("nan").cpu().numpy()
      pol["qpos"][k] = data.qpos.cpu().numpy()
      for key in ("q", "qd", "tau"):
        trace[key][s] = g[key][:, tr_t].cpu().numpy()
      trace["h"][s] = g["h"][:, tr_t].cpu().numpy()
  if lim > 0.0:
    assert check["max_step"] <= lim + 1e-5, check
    assert check["entry_max_from_measured"] <= lim + 1e-5, check
    assert check["entry_base_mismatch_max"] == 0.0, check
  assert check["raw_action_mismatch_max"] == 0.0, check
  t_roll = time.time() - t0

  os.makedirs(args.out, exist_ok=True)
  ckpt = os.path.abspath(args.checkpoint)
  import hashlib

  sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
  meta = {
    "checkpoint": ckpt,
    "checkpoint_sha256": sha,
    "iteration": int(torch.load(ckpt, map_location="cpu", weights_only=False)["iter"]),
    "limit_rad_per_step": lim,
    "limit_source": source,
    "config": "limited" if lim > 0 else "baseline",
    "task": args.task,
    "seed": args.seed,
    "per_pose": args.per_pose,
    "num_envs": n,
    "poses": list(FALLEN_POSE_NAMES),
    "physics_dt": env.physics_dt,
    "decimation": dec,
    "policy_dt": env.step_dt,
    "passive_steps": P,
    "episode_steps": T,
    "assist": None,
    "policy": "student, deterministic (mean action)",
    "joint_names": list(JOINT_NAMES),
    "q_min": qmin.cpu().tolist(),
    "q_max": qmax.cpu().tolist(),
    "kp": t._kp[0].cpu().tolist(),
    "kd": t._kd[0].cpu().tolist(),
    "tau_cap": TAU_CAP,
    "trace_envs": tr.tolist(),
    "origins": env.scene.env_origins.cpu().tolist(),
    "limiter_check": check,
    "rollout_seconds": t_roll,
  }
  init_qpos = q0.cpu().numpy()
  env.close()
  stats = run_stats(qd_u16, pol, pose_np, meta)
  meta["stats_seconds"] = time.time() - t0 - t_roll
  with open(os.path.join(args.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
  with open(os.path.join(args.out, "stats.json"), "w") as f:
    json.dump(stats["json"], f, indent=1)
  np.savez_compressed(
    os.path.join(args.out, "run.npz"),
    pose=pose_np,
    init_qpos=init_qpos,
    hist=stats["hist"],
    events=stats["events"],
    ph=stats["ph"],
    **pol,
    **{f"trace_{k}": v for k, v in trace.items()},
  )
  print(
    f"[limiter_eval] {args.out}: rollout {t_roll:.0f}s, "
    f"recovered {100 * stats['json']['overall']['recovered']:.1f}%",
    flush=True,
  )


# ---------------------------------------------------------------------------
# Per-run statistics


def phases(h: np.ndarray, up: np.ndarray, P: int, dt: float, first_inv: np.ndarray):
  """(T, n) phase per policy step: -1 passive, 0..3 = A..D, -2 invalid and after.

  A: before the first crossing of h1; B: to the first crossing of h2; C: to the start
  of the first 0.5 s run above h2 and upright; D: after (velocity-audit definition).
  """
  T, n = h.shape
  L = T - P
  w = int(round(STABLE_S / dt))
  ph = np.full((T, n), -1, np.int8)
  hh, uu = h[P:], up[P:]
  for e in range(n):
    a1 = hh[:, e] > H1
    k1 = int(a1.argmax()) if a1.any() else L
    k2 = L
    if k1 < L:
      a2 = hh[k1:, e] > H2
      k2 = k1 + int(a2.argmax()) if a2.any() else L
    ks = L
    if k2 < L:
      ok = (hh[k2:, e] > H2) & (uu[k2:, e] > STAND_UP)
      c = np.concatenate([[0], np.cumsum(ok)])
      run = (c[w:] - c[:-w]) == w
      ks = k2 + int(run.argmax()) if run.any() else L
    ph[P : P + k1, e] = 0
    ph[P + k1 : P + k2, e] = 1
    ph[P + k2 : P + ks, e] = 2
    ph[P + ks :, e] = 3
    if first_inv[e] < T:
      ph[first_inv[e] :, e] = -2
  return ph


def hist_quantile(hist: np.ndarray, q: float) -> float:
  c = np.cumsum(hist)
  if c[-1] == 0:
    return float("nan")
  return float(np.searchsorted(c, q / 100.0 * c[-1])) / Q


def hist_frac_above(hist: np.ndarray, thr: float) -> float:
  tot = hist.sum()
  return float(hist[int(round(thr * Q)) + 1 :].sum() / tot) if tot else float("nan")


def run_stats(qd_u16, pol, pose, meta) -> dict:
  dec, P, dt = meta["decimation"], meta["passive_steps"], meta["policy_dt"]
  T, n = pol["h"].shape
  nj = len(JOINT_NAMES)
  inv = pol["invalid"] | pol["nan_term"]
  first_inv = np.where(inv.any(0), inv.argmax(0), T)
  ph = phases(pol["h"], pol["up"], P, dt, first_inv)

  # Histograms (group, joint, bin) of |qdot| at every physics step.
  hist = np.zeros((len(GROUPS), nj, NB), np.int64)
  jj = np.arange(nj)[None, None, :]
  CH = 50  # policy steps per chunk
  for k0 in range(0, T, CH):
    k1 = min(T, k0 + CH)
    v = np.minimum(qd_u16[k0 * dec : k1 * dec].astype(np.int64), NB - 1)
    p = np.repeat(ph[k0:k1], dec, axis=0)[..., None]  # (s, n, 1)
    m = np.broadcast_to(p >= 0, v.shape)
    idx = (np.broadcast_to(p, v.shape) * nj + jj) * NB + v
    hist[:4] += np.bincount(idx[m], minlength=4 * nj * NB).reshape(4, nj, NB)
  # First 100 ms after the passive window (valid envs).
  s0, s1 = P * dec, P * dec + int(round(0.1 / meta["physics_dt"]))
  v = np.minimum(qd_u16[s0:s1].astype(np.int64), NB - 1)
  okenv = np.repeat(ph[P : P + (s1 - s0 + dec - 1) // dec], dec, 0)[: s1 - s0] >= 0
  m = np.broadcast_to(okenv[..., None], v.shape)
  idx = np.broadcast_to(jj, v.shape) * NB + v
  hist[4] += np.bincount(idx[m], minlength=nj * NB).reshape(nj, NB)

  # > 20 rad/s events (runs of consecutive physics steps), valid actuated samples.
  valid_sub = np.repeat(ph >= 0, dec, axis=0)  # (S, n)
  b = (qd_u16 > int(SPIKE * Q)) & valid_sub[..., None]
  pad = np.zeros((1, n, nj), bool)
  dd = np.diff(np.concatenate([pad, b, pad], 0).astype(np.int8), axis=0)
  s_start, e_id, j_id = np.nonzero(dd == 1)
  s_end, e2, j2 = np.nonzero(dd == -1)
  o1 = np.lexsort((s_start, j_id, e_id))
  o2 = np.lexsort((s_end, j2, e2))
  s_start, e_id, j_id, s_end = s_start[o1], e_id[o1], j_id[o1], s_end[o2]
  peak = np.array(
    [
      qd_u16[a:b_, e, j].max() / Q
      for a, b_, e, j in zip(s_start, s_end, e_id, j_id, strict=True)
    ],
    np.float32,
  )
  ev_phase = ph[s_start // dec, e_id] if len(s_start) else np.zeros(0, np.int8)
  events = np.rec.fromarrays(
    [e_id, j_id, s_start, s_end - s_start, peak, ev_phase],
    names="env,joint,start,len,peak,phase",
  )

  # Recovery (analyze_recovery.analyze, unchanged criterion).
  z = {
    "h": pol["h"].T,
    "up": pol["up"].T,
    "support": pol["support"].transpose(1, 0, 2),
    "passive": P,
    "dt": dt,
  }
  a = recovery_analyze(z)
  ok_after = ph >= 0
  vmax_ep = np.where(ok_after[..., None], pol["viol_step_max"].astype(np.float32), 0)
  ep = {
    "viol_max": vmax_ep.max((0, 2)),
    "qd_max": np.where(
      ok_after[..., None], pol["qd_step_max"].astype(np.float32), 0
    ).max((0, 2)),
    "tau_max": np.where(
      ok_after[..., None], pol["tau_step_max"].astype(np.float32), 0
    ).max((0, 2)),
    "err_max": np.where(
      ok_after[..., None], pol["err_step_max"].astype(np.float32), 0
    ).max((0, 2)),
    "root_f_max": np.where(ok_after, pol["root_f_peak"], 0).max(0) / MINIPI_WEIGHT,
    "events": np.bincount(e_id, minlength=n),
    "invalid": first_inv < T,
  }
  act_steps = ok_after.sum(0)
  sat = np.where(ok_after, pol["sat_count"], 0).sum(0)

  def summarize(sel: np.ndarray) -> dict:
    r = a["recovered"][sel]
    r1 = a["cross"]["h1"]["reached"][sel]
    r2 = a["cross"]["h2"]["reached"][sel]
    tr = a["t_rec"][sel & a["recovered"]]
    tr = tr[np.isfinite(tr)]
    vm = ep["viol_max"][sel]
    return {
      "episodes": int(sel.sum()),
      "recovered": float(r.mean()),
      "strict_0p31": float(a["strict"][sel].mean()),
      "foot_only_stable": float((a["recovered"] & a["final_feet_only"])[sel].mean()),
      "reach_h1": float(r1.mean()),
      "h2_given_h1": float(r2[r1].mean()) if r1.any() else 0.0,
      "recovered_given_h2": float(r[r2].mean()) if r2.any() else 0.0,
      "recover_time_median_s": float(np.median(tr)) if tr.size else None,
      "recover_time_p95_s": float(np.percentile(tr, 95)) if tr.size else None,
      "events_gt20_per_episode": float(ep["events"][sel].mean()),
      "viol_max": float(vm.max()),
      "viol_p99_episode": float(np.percentile(vm, 99)),
      "episodes_viol_gt_0p05": float((vm > VIOL).mean()),
      "tau_max": float(ep["tau_max"][sel].max()),
      "tau_sat_frac": float(sat[sel].sum() / max(1, act_steps[sel].sum() * dec * nj)),
      "track_err_max": float(ep["err_max"][sel].max()),
      "root_f_max_bw": float(ep["root_f_max"][sel].max()),
      "root_f_p99_episode_bw": float(np.percentile(ep["root_f_max"][sel], 99)),
      "episodes_root_f_gt_10bw": float((ep["root_f_max"][sel] > BIG_F).mean()),
      "invalid_episodes": int(ep["invalid"][sel].sum()),
      "categories": {
        c: float((a["cat"][sel] == c).mean()) for c in sorted(set(a["cat"][sel]))
      },
    }

  out = {
    "overall": summarize(np.ones(n, bool)),
    "per_pose": {
      name: summarize(pose == g) for g, name in enumerate(FALLEN_POSE_NAMES)
    },
  }
  ev_lens = events["len"] * meta["physics_dt"] * 1e3 if len(events) else np.zeros(0)
  out["events_gt20"] = {
    "count": int(len(events)),
    "mean_duration_ms": float(ev_lens.mean()) if len(events) else 0.0,
    "max_duration_ms": float(ev_lens.max()) if len(events) else 0.0,
    "per_phase": {GROUPS[p]: int((events["phase"] == p).sum()) for p in range(4)},
    "first_100ms": int(((events["start"] >= s0) & (events["start"] < s1)).sum()),
  }
  out["qd_max_true"] = float(qd_u16[valid_sub[..., None].repeat(nj, -1)].max() / Q)
  per_env = {
    "recovered": a["recovered"],
    "t_rec": a["t_rec"],
    "cat": a["cat"].astype(str),
    **ep,
  }
  out["per_env"] = {k: np.asarray(v).tolist() for k, v in per_env.items()}
  return {"json": out, "hist": hist, "events": events, "ph": ph}


# ---------------------------------------------------------------------------
# Report


def _load(root: str):
  runs = []
  for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
    if not os.path.exists(os.path.join(d, "stats.json")):
      continue
    m = json.load(open(os.path.join(d, "meta.json")))
    s = json.load(open(os.path.join(d, "stats.json")))
    runs.append({"dir": d, "meta": m, "stats": s})
  return runs


def _key(r):
  return (r["meta"]["iteration"], r["meta"]["config"])


def _hist(r):
  return np.load(os.path.join(r["dir"], "run.npz"))["hist"]


def _hmax(h: np.ndarray) -> float:
  nz = np.nonzero(h)[0]
  return float(nz.max()) / Q if nz.size else float("nan")


def _csv(path, rows):
  if not rows:
    return
  keys = list(rows[0])
  for r in rows[1:]:
    keys += [k for k in r if k not in keys]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)


def _fmt(x, nd=3):
  return None if x is None else round(float(x), nd)


def report(args) -> None:
  root = args.root
  runs = _load(root)
  its = sorted({r["meta"]["iteration"] for r in runs})
  seeds = sorted({r["meta"]["seed"] for r in runs})
  cfgs = ("baseline", "limited")

  # Pairing check: identical initial states for every (iteration, seed) pair, and
  # identical across checkpoints for a seed.
  init = {}
  for r in runs:
    q0 = np.load(os.path.join(r["dir"], "run.npz"))["init_qpos"]
    init.setdefault(r["meta"]["seed"], []).append(q0)
  pairing = {
    str(s): float(max(np.abs(x - v[0]).max() for x in v)) for s, v in init.items()
  }

  per_pose_rows, comp_rows, jv_rows, seed_rows = [], [], [], []
  pooled = {}
  for it in its:
    for c in cfgs:
      rs = [r for r in runs if _key(r) == (it, c)]
      if not rs:
        continue
      # Pool episodes over seeds (per-env lists).
      pe = {}
      for r in rs:
        for k, v in r["stats"]["per_env"].items():
          pe.setdefault(k, []).extend(v)
      pe = {k: np.array(v) for k, v in pe.items()}
      pose = np.concatenate(
        [np.load(os.path.join(r["dir"], "run.npz"))["pose"] for r in rs]
      )
      hist = sum(_hist(r) for r in rs)
      ev = sum(r["stats"]["events_gt20"]["count"] for r in rs)
      n_ep = len(pose)
      for r in rs:
        o = r["stats"]["overall"]
        seed_rows.append(
          {
            "iteration": it,
            "config": c,
            "seed": r["meta"]["seed"],
            "recovered": _fmt(o["recovered"]),
            "worst_pose": _fmt(
              min(v["recovered"] for v in r["stats"]["per_pose"].values())
            ),
            "events_gt20_per_episode": _fmt(o["events_gt20_per_episode"], 2),
            "invalid_episodes": o["invalid_episodes"],
          }
        )
        for name, v in r["stats"]["per_pose"].items():
          per_pose_rows.append(
            {
              "iteration": it,
              "config": c,
              "seed": r["meta"]["seed"],
              "pose": name,
              **{
                k: (_fmt(x) if not isinstance(x, dict) else None)
                for k, x in v.items()
                if k != "categories"
              },
              "failure_categories": "; ".join(
                f"{k} {100 * x:.1f}%"
                for k, x in v["categories"].items()
                if k != "recovered"
              ),
            }
          )
      # Pooled per pose.
      rec_pose = {
        name: float(pe["recovered"][pose == g].mean())
        for g, name in enumerate(FALLEN_POSE_NAMES)
      }
      tr = pe["t_rec"][pe["recovered"].astype(bool)]
      tr = tr[np.isfinite(tr.astype(float))].astype(float)
      vm = pe["viol_max"].astype(float)
      seed_rec = [r["stats"]["overall"]["recovered"] for r in rs]
      seed_worst = [
        min(v["recovered"] for v in r["stats"]["per_pose"].values()) for r in rs
      ]
      rf = pe["root_f_max"].astype(float)
      row = {
        "checkpoint": f"model_{it}",
        "limiter": "0.3 rad/step" if c == "limited" else "none",
        "seeds": "/".join(str(r["meta"]["seed"]) for r in rs),
        "episodes": n_ep,
        "recovery": _fmt(pe["recovered"].mean()),
        "recovery_min_seed": _fmt(min(seed_rec)),
        "worst_pose": _fmt(min(rec_pose.values())),
        "worst_pose_min_seed": _fmt(min(seed_worst)),
        "worst_pose_name": min(rec_pose, key=rec_pose.get),
        "recovery_time_median_s": _fmt(np.median(tr)) if tr.size else None,
        "recovery_time_p95_s": _fmt(np.percentile(tr, 95)) if tr.size else None,
        "lifting_qd_p95": _fmt(hist_quantile(hist[0].sum(0), 95), 2),
        "lifting_qd_p99": _fmt(hist_quantile(hist[0].sum(0), 99), 2),
        "lifting_qd_max": _fmt(_hmax(hist[0].sum(0)), 1),
        "qd_p99_actuated": _fmt(hist_quantile(hist[:4].sum((0, 1)), 99), 2),
        "qd_max": _fmt(max(r["stats"]["qd_max_true"] for r in rs), 1),
        "frac_gt_4": _fmt(hist_frac_above(hist[:4].sum((0, 1)), 4.0), 4),
        "frac_gt_6p28": _fmt(hist_frac_above(hist[:4].sum((0, 1)), 6.28), 4),
        "frac_gt_10": _fmt(hist_frac_above(hist[:4].sum((0, 1)), 10.0), 5),
        "frac_gt_20": _fmt(hist_frac_above(hist[:4].sum((0, 1)), 20.0), 6),
        "events_gt20": int(ev),
        "events_gt20_per_episode": _fmt(ev / n_ep, 2),
        "events_gt20_first_100ms": sum(
          r["stats"]["events_gt20"]["first_100ms"] for r in rs
        ),
        "joint_violation_max": _fmt(vm.max()),
        "joint_violation_p99_episode": _fmt(np.percentile(vm, 99)),
        "episodes_violation_gt_0p05": _fmt((vm > VIOL).mean()),
        "tau_max": _fmt(pe["tau_max"].astype(float).max(), 2),
        "tau_sat_frac": _fmt(
          np.average(
            [r["stats"]["overall"]["tau_sat_frac"] for r in rs],
          ),
          5,
        ),
        "track_err_max": _fmt(pe["err_max"].astype(float).max()),
        "root_force_max_bw": _fmt(rf.max(), 1),
        "episodes_root_force_gt_10bw": _fmt((rf > BIG_F).mean()),
        "invalid_episodes": int(pe["invalid"].astype(bool).sum()),
      }
      for name in FALLEN_POSE_NAMES:
        row[f"recovery_{name}"] = _fmt(rec_pose[name])
      pooled[(it, c)] = {"row": row, "hist": hist, "pose": pose, "pe": pe}
      comp_rows.append(row)
      # Joint-velocity statistics (pooled seeds).
      for gi, gname in enumerate(GROUPS):
        for j in range(-1, len(JOINT_NAMES)):
          hh = hist[gi].sum(0) if j < 0 else hist[gi, j]
          jv_rows.append(
            {
              "checkpoint": f"model_{it}",
              "limiter": row["limiter"],
              "scope": gname,
              "joint": "all" if j < 0 else SHORT[j],
              "samples": int(hh.sum()),
              "p95": _fmt(hist_quantile(hh, 95), 2),
              "p99": _fmt(hist_quantile(hh, 99), 2),
              "max": _fmt(
                (np.nonzero(hh)[0].max() / Q) if hh.sum() else float("nan"), 2
              ),
              **{
                f"frac_gt_{str(x).replace('.', 'p')}": _fmt(hist_frac_above(hh, x), 5)
                for x in THRESH
              },
            }
          )
      # Events per joint / phase (pooled).
      evs = np.concatenate(
        [np.load(os.path.join(r["dir"], "run.npz"))["events"] for r in rs]
      )
      for j in range(len(JOINT_NAMES)):
        e = evs[evs["joint"] == j]
        jv_rows.append(
          {
            "checkpoint": f"model_{it}",
            "limiter": row["limiter"],
            "scope": "events_gt20",
            "joint": SHORT[j],
            "samples": int(len(e)),
            "event_mean_duration_ms": _fmt(e["len"].mean() * 0.5, 2)
            if len(e)
            else None,
            "event_max_duration_ms": _fmt(e["len"].max() * 0.5, 2) if len(e) else None,
          }
        )
  # Reductions vs each checkpoint's own baseline.
  for row in comp_rows:
    it = int(row["checkpoint"].split("_")[1])
    b = pooled.get((it, "baseline"))
    if b is None:
      continue
    br = b["row"]
    for k in ("events_gt20", "lifting_qd_p99", "qd_max", "joint_violation_max"):
      x, y = row[k], br[k]
      row[f"{k}_change_vs_baseline_pct"] = _fmt(100.0 * (x - y) / y, 1) if y else None
    if br["recovery_time_median_s"] and row["recovery_time_median_s"]:
      row["recovery_time_median_delta_s"] = _fmt(
        row["recovery_time_median_s"] - br["recovery_time_median_s"], 2
      )
  # Changes vs a reference row (e.g. the source checkpoint with the same limiter).
  ref = getattr(args, "ref_iteration", 0)
  rr = pooled.get((ref, "limited"), {}).get("row") if ref else None
  if rr is not None:
    for row in comp_rows:
      for k in (
        "recovery",
        "worst_pose",
        "events_gt20_per_episode",
        "lifting_qd_p99",
        "qd_max",
        "joint_violation_max",
        "episodes_violation_gt_0p05",
        "recovery_time_median_s",
        "recovery_time_p95_s",
      ):
        x, y = row[k], rr[k]
        row[f"{k}_delta_vs_ref"] = (
          _fmt(x - y, 4) if x is not None and y is not None else None
        )
  os.makedirs(root, exist_ok=True)
  _csv(os.path.join(root, "checkpoint_comparison.csv"), comp_rows)
  _csv(os.path.join(root, "per_pose_results.csv"), per_pose_rows)
  _csv(os.path.join(root, "per_seed_results.csv"), seed_rows)
  _csv(os.path.join(root, "joint_velocity_statistics.csv"), jv_rows)
  checks = {
    os.path.basename(r["dir"]): {
      "limiter_check": r["meta"]["limiter_check"],
      "rollout_seconds": round(r["meta"]["rollout_seconds"]),
    }
    for r in runs
  }
  with open(os.path.join(root, "analysis.json"), "w") as f:
    json.dump(
      {
        "iterations": its,
        "seeds": seeds,
        "init_state_max_abs_diff_per_seed": pairing,
        "runs": checks,
      },
      f,
      indent=1,
    )
  plots(root, runs, pooled, its)
  print("[report] done", flush=True)


# ---------------------------------------------------------------------------
# Plots


def _style():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {
      "figure.dpi": 110,
      "font.size": 9,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "grid.color": "#e3e3e3",
      "grid.linewidth": 0.6,
    }
  )
  return plt


C_BASE, C_LIM = "#5b6b7a", "#d0731f"


def plots(root, runs, pooled, its) -> None:
  """One compact summary figure (``plots/report.png``)."""
  del runs
  plt = _style()
  pdir = os.path.join(root, "plots")
  os.makedirs(pdir, exist_ok=True)
  present = [c for c in ("baseline", "limited") if any(k[1] == c for k in pooled)]
  col = {"baseline": C_BASE, "limited": C_LIM}
  lab = {"baseline": "no limiter", "limited": "limiter 0.3 rad/step"}
  x = np.arange(len(its))
  w = 0.8 / len(present)
  panels = (
    ("recovery", "Recovery, pooled (%)", 100.0, "{:.1f}"),
    ("worst_pose_min_seed", "Worst pose, min over seeds (%)", 100.0, "{:.1f}"),
    ("events_gt20_per_episode", "|qdot| > 20 rad/s events / episode", 1.0, "{:.2f}"),
    ("qd_max", "Max |qdot| (rad/s)", 1.0, "{:.1f}"),
    ("joint_violation_max", "Max joint-limit penetration (rad)", 1.0, "{:.2f}"),
    ("recovery_time_median_s", "Median recovery time (s)", 1.0, "{:.2f}"),
  )
  fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
  for ax, (key, title, sc, fmt) in zip(axs.flat, panels, strict=True):
    for k, c in enumerate(present):
      v = [
        (pooled[(i, c)]["row"][key] or 0) * sc if (i, c) in pooled else np.nan
        for i in its
      ]
      off = (k - (len(present) - 1) / 2) * w
      b = ax.bar(x + off, v, w * 0.92, color=col[c], label=lab[c])
      ax.bar_label(b, [fmt.format(y) for y in v], fontsize=7, padding=1)
    ax.set_xticks(x, [f"{i}" for i in its], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.margins(y=0.15)
  axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
  fig.suptitle(
    f"{os.path.basename(os.path.normpath(root))}: deterministic student, zero assist, "
    "seeds pooled (x-axis: checkpoint iteration)",
    fontsize=10,
  )
  fig.tight_layout()
  fig.savefig(os.path.join(pdir, "report.png"))
  plt.close(fig)


def _trace_pair(root, it, seed, e_local, out):
  """Paired baseline / limited traces of one trace env, identical axes."""
  plt = _style()
  R = {}
  for c in ("baseline", "limited"):
    d = os.path.join(root, "runs", f"it{it}_{c}_s{seed}")
    R[c] = (json.load(open(os.path.join(d, "meta.json"))), np.load(f"{d}/run.npz"))
  meta, z = R["baseline"]
  e = meta["trace_envs"][e_local]
  dec, P, pdt = meta["decimation"], meta["passive_steps"], meta["policy_dt"]
  sdt = meta["physics_dt"]
  zb = R["baseline"][1]
  j = int(np.abs(zb["trace_qd"][P * dec :, e_local]).max(0).argmax())
  qmin, qmax = meta["q_min"][j], meta["q_max"][j]
  t0 = P * pdt - 0.1
  t1 = P * pdt + 3.0
  fig, axs = plt.subplots(5, 2, figsize=(10, 9.5), sharex=True, sharey="row")
  for col, c in enumerate(("baseline", "limited")):
    m, zz = R[c]
    S = zz["trace_qd"].shape[0]
    ts = (np.arange(S) + 1) * sdt
    tp = (np.arange(zz["q_star"].shape[0]) + 1) * pdt
    ax = axs[0, col]
    ax.plot(ts, zz["trace_qd"][:, e_local, j], color=C_LIM if col else C_BASE, lw=0.8)
    ax.axhline(20, color="#999", lw=0.6, ls="--")
    ax.axhline(-20, color="#999", lw=0.6, ls="--")
    ax.set_title(
      f"{'no limiter' if c == 'baseline' else 'limiter 0.3 rad/step'} - "
      f"model_{it} env {e} {FALLEN_POSE_NAMES[int(zz['pose'][e])]}, {SHORT[j]}"
    )
    ax.set_ylabel("qdot (rad/s)")
    ax = axs[1, col]
    ax.step(
      tp, zz["q_star"][:, e, j], where="post", color="#1f6fb2", lw=1.0, label="q*"
    )
    ax.plot(ts, zz["trace_q"][:, e_local, j], color="#222", lw=0.9, label="q")
    ax.axhline(qmin, color="#c33", lw=0.6, ls=":")
    ax.axhline(qmax, color="#c33", lw=0.6, ls=":")
    ax.set_ylabel("q (rad)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axs[2, col]
    ax.plot(ts, zz["trace_tau"][:, e_local, j], color="#555", lw=0.7)
    ax.axhline(TAU_CAP, color="#c33", lw=0.6, ls=":")
    ax.axhline(-TAU_CAP, color="#c33", lw=0.6, ls=":")
    ax.set_ylabel("tau (Nm)")
    ax = axs[3, col]
    ax.plot(ts, zz["trace_h"][:, e_local], color="#333", lw=1.0)
    ax.axhline(H1, color="#2a9d8f", lw=0.7, ls="--", label="h1")
    ax.axhline(H2, color="#e76f51", lw=0.7, ls="--", label="h2")
    ax.set_ylabel("base h (m)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axs[4, col]
    v = zz["viol_step_max"][:, e].astype(np.float32).max(-1)
    ax.step(tp, v, where="post", color="#c33", lw=0.9)
    ax.set_ylabel("max limit\npenetration (rad)")
    ax.set_xlabel("time (s)")
  axs[0, 0].set_xlim(t0, t1)
  fig.tight_layout()
  fig.savefig(out)
  plt.close(fig)
  return {"env": int(e), "joint": SHORT[j]}


def videos(args) -> None:
  import imageio.v2 as imageio

  from minipi_getup.ftsr_ref.analyze_recovery import _scene_model
  from minipi_getup.ftsr_ref.visual_demo import Renderer

  root = args.root
  it, seed = args.iteration, args.seed
  rend = Renderer(_scene_model())
  vdir = os.path.join(root, "videos")
  pdir = os.path.join(root, "plots")
  os.makedirs(vdir, exist_ok=True)
  R = {}
  for c in ("baseline", "limited"):
    d = os.path.join(root, "runs", f"it{it}_{c}_s{seed}")
    R[c] = {
      "meta": json.load(open(os.path.join(d, "meta.json"))),
      "stats": json.load(open(os.path.join(d, "stats.json"))),
      "z": np.load(os.path.join(d, "run.npz")),
    }
  meta = R["limited"]["meta"]
  dt, P = meta["policy_dt"], meta["passive_steps"]
  origins = np.array(meta["origins"])
  pe_l = R["limited"]["stats"]["per_env"]
  pe_b = R["baseline"]["stats"]["per_env"]
  rec_l = np.array(pe_l["recovered"], bool)
  tr_env = meta["trace_envs"]
  sel = {}
  # Success with limiter: a trace env (full traces) that recovers in both configs.
  ok_tr = [k for k, e in enumerate(tr_env) if rec_l[e] and pe_b["recovered"][e]]
  if ok_tr:
    sel["success_limited"] = tr_env[ok_tr[0]]
  # Failure with limiter (worst pose first).
  fail = np.nonzero(~rec_l)[0]
  if fail.size:
    sel["failed_limited"] = int(fail[0])
  # Paired baseline vs limiter: trace env with the largest baseline peak.
  qmb = np.array(pe_b["qd_max"])[tr_env]
  k_pair = int(np.argmax(qmb))
  sel["paired"] = tr_env[k_pair]
  info = {}
  for tag, e in sel.items():
    pose = FALLEN_POSE_NAMES[int(R["limited"]["z"]["pose"][e])]
    cfgs = ("baseline", "limited") if tag == "paired" else ("limited",)
    frames = []
    T = R["limited"]["z"]["qpos"].shape[0]
    for k in range(T):
      imgs = []
      for c in cfgs:
        z = R[c]["z"]
        rend.lookat = None if k == 0 else rend.lookat
        img = rend.frame(z["qpos"][k, e], origins[e])
        qm = float(z["qd_step_max"][k, e].astype(np.float32).max())
        cls = (
          "recovered" if R[c]["stats"]["per_env"]["recovered"][e] else "not recovered"
        )
        lines = [
          f"model_{it} env {e} {pose} seed {seed}",
          f"{'no limiter' if c == 'baseline' else 'limiter 0.3 rad/step'} [{cls}]",
          f"t = {(k + 1) * dt:5.2f} s{'  (passive)' if k < P else ''}",
          f"base height {z['h'][k, e]:.3f} m (h1 0.19, h2 0.276)",
          f"max |qdot| in step {qm:5.1f} rad/s",
          "zero assistance, deterministic student",
        ]
        imgs.append(rend.overlay(img, lines))
      frames.append(np.concatenate(imgs, axis=1) if len(imgs) > 1 else imgs[0])
    path = os.path.join(vdir, f"it{it}_{tag}_env{e}_realtime.mp4")
    imageio.mimwrite(path, frames, fps=int(round(1.0 / dt)), quality=8)
    slow = os.path.join(vdir, f"it{it}_{tag}_env{e}_slow0p25x.mp4")
    k_end = min(T, P + int(4.0 / dt))
    imageio.mimwrite(slow, frames[P - 5 : k_end], fps=12, quality=8)
    info[tag] = {"env": int(e), "pose": pose, "realtime": path, "slow_0p25x": slow}
  # Paired trace plots (trace envs, one per pose), identical axes per pair.
  plots_ = {}
  for k in range(0, len(tr_env), max(1, len(tr_env) // 4)):
    out = os.path.join(pdir, f"paired_trace_it{it}_env{tr_env[k]}.png")
    plots_[out] = _trace_pair(root, it, seed, k, out)
  info["paired_trace_plots"] = plots_
  with open(os.path.join(vdir, f"videos_it{it}.json"), "w") as f:
    json.dump(info, f, indent=1)
  print("[videos] done", flush=True)


def videos_selected(args) -> None:
  """ONE real-time video of one run: a 2 x 2 grid of the four poses, each the lowest
  env index of that pose that recovered (fixed rule, not chosen by appearance), from
  the passive window to the end of the episode. If the run has failures, the
  lowest-index failure replaces nothing and is written as a second video."""
  import imageio.v2 as imageio

  from minipi_getup.ftsr_ref.analyze_recovery import _scene_model
  from minipi_getup.ftsr_ref.visual_demo import Renderer

  meta = json.load(open(os.path.join(args.run, "meta.json")))
  st = json.load(open(os.path.join(args.run, "stats.json")))
  z = np.load(os.path.join(args.run, "run.npz"))
  rec = np.array(st["per_env"]["recovered"], bool)
  cat = st["per_env"]["cat"]
  pose = z["pose"]
  grid = []
  for g in range(len(FALLEN_POSE_NAMES)):
    ok = np.nonzero((pose == g) & rec)[0]
    if ok.size:
      grid.append(int(ok[0]))
  bad = np.nonzero(~rec)[0]
  rends = [Renderer(_scene_model()) for _ in grid]
  os.makedirs(args.out, exist_ok=True)
  dt, P = meta["policy_dt"], meta["passive_steps"]
  origins = np.array(meta["origins"])
  T = z["qpos"].shape[0]
  tag0 = f"{args.label or 'it' + str(meta['iteration'])}_s{meta['seed']}"

  def tile(rend, e, k):
    img = rend.frame(z["qpos"][k, e], origins[e])
    qm = float(z["qd_step_max"][k, e].astype(np.float32).max())
    lines = [
      f"{FALLEN_POSE_NAMES[int(pose[e])]} env {e} "
      f"[{'recovered' if rec[e] else 'NOT recovered: ' + cat[e]}]",
      f"t {(k + 1) * dt:5.2f} s{' passive' if k < P else ''}  h {z['h'][k, e]:.3f} m",
      f"max |qdot| {qm:4.1f} rad/s",
    ]
    return rend.overlay(img, lines)

  head = (
    f"{tag0}  limiter {meta['limit_rad_per_step']} rad/step on the PD target "
    "(joint speed not hard-limited), zero assist, deterministic, simulation"
  )
  frames = []
  for k in range(T):
    t_ = [tile(r, e, k) for r, e in zip(rends, grid, strict=True)]
    while len(t_) < 4:
      t_.append(np.zeros_like(t_[0]))
    img = np.concatenate(
      [np.concatenate(t_[:2], axis=1), np.concatenate(t_[2:], axis=1)], axis=0
    )
    if k == 0:
      strip = rends[0].overlay(np.zeros((38,) + img.shape[1:], np.uint8), [head])
    frames.append(np.concatenate([strip, img], axis=0))
  out = {"grid": os.path.join(args.out, f"{tag0}_4poses_realtime.mp4")}
  imageio.mimwrite(out["grid"], frames, fps=int(round(1.0 / dt)), quality=7)
  if bad.size:
    e = int(bad[0])
    rend = Renderer(_scene_model())
    out["failure"] = os.path.join(args.out, f"{tag0}_failure_env{e}_realtime.mp4")
    imageio.mimwrite(
      out["failure"],
      [tile(rend, e, k) for k in range(T)],
      fps=int(round(1.0 / dt)),
      quality=7,
    )
  info = {
    "selection_rule": "per pose: lowest env index that recovered; failure: lowest "
    "env index that did not recover",
    "grid_envs": dict(zip(FALLEN_POSE_NAMES, grid, strict=False)),
    "failure_env": int(bad[0]) if bad.size else None,
    "checkpoint": meta["checkpoint"],
    "checkpoint_sha256": meta["checkpoint_sha256"],
    "seed": meta["seed"],
    "limiter": f"{meta['limit_rad_per_step']} rad/policy step ({meta['limit_source']})",
    "pose_recovery": {k: v["recovered"] for k, v in st["per_pose"].items()},
    "videos": out,
  }
  with open(os.path.join(args.out, f"{tag0}_videos.json"), "w") as f:
    json.dump(info, f, indent=1)
  print("[videos_selected] done", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  sp = ap.add_subparsers(dest="cmd", required=True)
  r = sp.add_parser("rollout")
  r.add_argument("--checkpoint", required=True)
  r.add_argument("--out", required=True)
  r.add_argument("--limit", type=float, default=0.0)
  r.add_argument(
    "--task", default=TASK, help="a task with a cfg limiter needs --limit 0"
  )
  r.add_argument("--seed", type=int, default=2150)
  r.add_argument("--per-pose", type=int, default=128)
  r.add_argument("--trace-per-pose", type=int, default=2)
  r.add_argument("--steps", type=int, default=0, help="debug: policy steps (0 = all)")
  a = sp.add_parser("report")
  a.add_argument("--root", required=True)
  a.add_argument("--ref-iteration", type=int, default=0)
  v = sp.add_parser("videos")
  v.add_argument("--root", required=True)
  v.add_argument("--iteration", type=int, required=True)
  v.add_argument("--seed", type=int, default=2150)
  vs = sp.add_parser("videos_selected")
  vs.add_argument("--run", required=True)
  vs.add_argument("--out", required=True)
  vs.add_argument("--label", default="")
  args = ap.parse_args()
  {
    "rollout": rollout,
    "report": report,
    "videos": videos,
    "videos_selected": videos_selected,
  }[args.cmd](args)


if __name__ == "__main__":
  main()
