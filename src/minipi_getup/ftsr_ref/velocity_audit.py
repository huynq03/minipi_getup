"""Joint-velocity audit of FTSR recovery checkpoints (evaluation only, simulation).

Step 1, one process per checkpoint (same seed and reset states for every checkpoint):

  python -m minipi_getup.ftsr_ref.velocity_audit rollout --checkpoint CKPT \
      --out DIR [--per-pose 32] [--seed 2150]

  Play cfg (no observation noise, ZERO assistance), deterministic student, fallen
  poses by env index, one 20 s episode per env (2 s passive + 18 s policy), the
  rollout of ``analyze_recovery``/``evaluate``. Recorded at every 0.5 ms physics step
  (hook around ``sim.step``): q, qdot, actuator force, unclipped PD torque (from the
  pre-step state), joint-space constraint force (contacts + joint limits), base
  height / uprightness, ground-contact groups. Per 20 ms policy step: q*, the
  unclipped target, the action, qpos (videos), each reward term, validity flags.

  The stage of the reward table is fixed to r_w (stage 2) during the audit: the
  training population is in r_w (S1/S2 ~ 0.87 at it 3000-3700), so the logged
  rewards are the ones the policy is trained on. No assistance in play, so the stage
  changes nothing else (dynamics and actions are unaffected).

Step 2: ``analyze --runs DIR... --out OUT`` writes the CSV tables, plots and JSON
summaries; ``videos --run DIR --out OUT`` renders selected recorded trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

import minipi_getup  # noqa: F401
from minipi_getup.ftsr_ref.config.env_cfg import STAGE_HEIGHTS
from minipi_getup.ftsr_ref.config.robot import JOINT_NAMES, TAU_CAP
from minipi_getup.ftsr_ref.evaluate import (
  DEV,
  STAND_UP,
  SUPPORT_GROUPS,
  SupportTracker,
  build_env,
)
from minipi_getup.ftsr_ref.export import load_model
from minipi_getup.ftsr_ref.mdp.actions import INVALID_KEY
from minipi_getup.ftsr_ref.mdp.resets import FALLEN_POSE_NAMES, POSE_KEY
from minipi_getup.ftsr_ref.rl.modules import StudentPolicy

TASK = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless"
H1, H2 = STAGE_HEIGHTS[0], STAGE_HEIGHTS[1]
H_SUCCESS = 0.9 * STAGE_HEIGHTS[2]
LAST_S = 3.0
STABLE_S = 0.5  # phase D starts after 0.5 s continuously above h2 and upright
SHORT = [x.replace("_joint", "") for x in JOINT_NAMES]
PHASES = ("A_lift", "B_h1_to_h2", "C_stabilize", "D_standing")
THRESH = (4.0, 6.28, 10.0, 20.0)
# Terms that penalize joint speed or its rate of change.
VEL_TERMS = (
  "qd_soft_envelope",
  "pen_dof_vel_l2",
  "pen_dof_acc_l2",
  "pen_joint_power_l2",
)


# ---------------------------------------------------------------------------
# Rollout


def rollout(args) -> None:
  model = load_model(args.checkpoint).to(DEV)
  npose = len(FALLEN_POSE_NAMES)
  n = args.per_pose * npose

  def mutate(cfg):
    cfg.events["reset_pose"].params["by_env_index"] = True
    cfg.seed = args.seed
    cfg.events["stage_setup"].params["stage"].fixed_stage = 2

  torch.manual_seed(args.seed)
  env = build_env(TASK, n, mutate)
  t = env.action_manager.get_term("joint_pos")
  robot = env.scene["robot"]
  mm = env.sim.mj_model
  import mujoco

  dof = torch.tensor(
    [
      mm.jnt_dofadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]
      for j in JOINT_NAMES
    ],
    device=DEV,
  )
  support = SupportTracker(env)
  dec = int(env.cfg.decimation)
  T = int(env.max_episode_length) - 1
  S = T * dec
  nj = len(JOINT_NAMES)
  pose = env.extras[POSE_KEY].clone()
  origin_z = env.scene.env_origins[:, 2]
  data = env.sim.data

  # Host arrays (time-major).
  sub = {
    "q": np.zeros((S, n, nj), np.float32),
    "qd": np.zeros((S, n, nj), np.float32),
    "tau": np.zeros((S, n, nj), np.float16),
    "tau_pd": np.zeros((S, n, nj), np.float16),
    "qfrc_con": np.zeros((S, n, nj), np.float16),
    "h": np.zeros((S, n), np.float32),
    "up": np.zeros((S, n), np.float16),
    "touch": np.zeros((S, n), np.uint8),  # bit g = SUPPORT_GROUPS[g] touching
  }
  pol = {
    "q_star": np.zeros((T, n, nj), np.float32),
    "q_cmd_unclipped": np.zeros((T, n, nj), np.float32),
    "action": np.zeros((T, n, nj), np.float32),
    "qpos": np.zeros((T, n, data.qpos.shape[1]), np.float32),
    "invalid": np.zeros((T, n), bool),
    "nan_term": np.zeros((T, n), bool),
  }
  rm = env.reward_manager
  names = list(rm.active_terms)
  pol["reward"] = np.zeros((T, n, len(names)), np.float32)

  # Per-policy-step device buffers.
  g = {
    k: torch.zeros((dec,) + v.shape[1:], device=DEV, dtype=torch.float32)
    for k, v in sub.items()
  }
  bit = (2 ** torch.arange(len(SUPPORT_GROUPS), device=DEV)).float()
  state = {"j": 0}
  step = env.sim.step

  def hooked():
    j = state["j"]
    q0 = robot.data.joint_pos[:, t._ids]
    qd0 = robot.data.joint_vel[:, t._ids]
    g["tau_pd"][j] = t._kp_eff * (t.q_star - q0) - t._kd_eff * qd0
    g["touch"][j] = support.touching().float() @ bit
    step()
    g["q"][j] = robot.data.joint_pos[:, t._ids]
    g["qd"][j] = robot.data.joint_vel[:, t._ids]
    g["tau"][j] = data.actuator_force[:, t._ctrl]
    g["qfrc_con"][j] = data.qfrc_constraint[:, dof]
    qp = data.qpos
    g["h"][j] = qp[:, 2] - origin_z
    x, y = qp[:, 4], qp[:, 5]
    g["up"][j] = 1.0 - 2.0 * (x * x + y * y)
    state["j"] = j + 1

  env.sim.step = hooked
  if args.target_rate_limit > 0.0:
    # COUNTERFACTUAL (evaluation only, not the training plant): limit the PD target
    # change to +-target_rate_limit rad per policy step, starting from the measured
    # pose at the end of the passive window.
    lim = float(args.target_rate_limit)
    orig = t.process_actions
    prev = {"q": None}

    def limited(actions):
      orig(actions)
      q_meas = torch.clamp(robot.data.joint_pos[:, t._ids], t.q_min, t.q_max)
      base = (
        q_meas
        if prev["q"] is None
        else torch.where(t.passive.unsqueeze(-1), q_meas, prev["q"])
      )
      t.q_star = base + torch.clamp(t.q_star - base, -lim, lim)
      prev["q"] = t.q_star.clone()

    t.process_actions = limited
  policy = StudentPolicy(model).to(DEV).eval()
  obs = env.get_observations()
  tm = env.termination_manager
  with torch.no_grad():
    for k in range(T):
      a = policy(obs["policy"])
      state["j"] = 0
      obs, *_ = env.step(a)
      assert state["j"] == dec
      s = slice(k * dec, (k + 1) * dec)
      for key, buf in g.items():
        sub[key][s] = buf.cpu().numpy().astype(sub[key].dtype)
      ac = torch.clamp(a, -t.cfg.raw_clip, t.cfg.raw_clip)
      pol["q_star"][k] = t.q_star.cpu().numpy()
      pol["q_cmd_unclipped"][k] = (t.default + t.scale * ac).cpu().numpy()
      pol["action"][k] = a.cpu().numpy()
      pol["qpos"][k] = data.qpos.cpu().numpy()
      pol["invalid"][k] = env.extras[INVALID_KEY].cpu().numpy()
      pol["nan_term"][k] = tm.get_term("nan").cpu().numpy()
      pol["reward"][k] = rm._step_reward.cpu().numpy()
  os.makedirs(args.out, exist_ok=True)
  ckpt = os.path.abspath(args.checkpoint)
  meta = {
    "checkpoint": ckpt,
    "iteration": int(torch.load(ckpt, map_location="cpu", weights_only=False)["iter"]),
    "task": TASK,
    "seed": args.seed,
    "per_pose": args.per_pose,
    "num_envs": n,
    "poses": list(FALLEN_POSE_NAMES),
    "physics_dt": env.physics_dt,
    "decimation": dec,
    "policy_dt": env.step_dt,
    "passive_steps": int(t.cfg.passive_steps),
    "episode_steps": T,
    "assist": None,
    "counterfactual_target_rate_limit_rad_per_step": args.target_rate_limit or None,
    "policy": "student, deterministic (mean action)",
    "reward_stage_fixed": 2,
    "reward_terms": names,
    "joint_names": list(JOINT_NAMES),
    "q_min": t.q_min.cpu().tolist(),
    "q_max": t.q_max.cpu().tolist(),
    "kp": t._kp[0].cpu().tolist(),
    "kd": t._kd[0].cpu().tolist(),
    "tau_cap": TAU_CAP,
    "support_groups": list(SUPPORT_GROUPS),
    "origins": env.scene.env_origins.cpu().tolist(),
  }
  with open(os.path.join(args.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
  np.savez(os.path.join(args.out, "rollout.npz"), pose=pose.cpu().numpy(), **sub, **pol)
  env.close()
  print(f"[velocity_audit] rollout done: {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Analysis

SAT = 0.95 * TAU_CAP  # "at the torque cap"
PEN_EPS = 0.005  # rad, physical joint-limit penetration counted as a violation
SPIKE = 20.0  # rad/s, spike events
# A target step of 0.25 rad within one 20 ms policy step commands >= 12.5 rad/s on
# its own (2x the 6.28 rad/s envelope): "large target jump".
JUMP = 0.25
LEG = (tuple(range(0, 6)), tuple(range(6, 12)))
CAUSES = {
  1: "1 large policy target jump (>= 0.25 rad / 20 ms, toward the motion)",
  2: "2 actuator-driven without a target jump (sustained error / saturation)",
  3: "3 ground impact / contact impulse",
  4: "4 joint-limit interaction",
  5: "5 physics instability / explosion",
  6: "6 other (inertial / gravity coupling) or unresolved",
}


def _runs(b: np.ndarray):
  """b: (n, nj, S) bool -> (env, joint, start, end) of True runs (end exclusive)."""
  pad = np.zeros(b.shape[:2] + (1,), bool)
  d = np.diff(np.concatenate([pad, b, pad], -1).astype(np.int8), axis=-1)
  e0, j0, s0 = np.nonzero(d == 1)
  _, _, s1 = np.nonzero(d == -1)
  return e0, j0, s0, s1


def _pct(x: np.ndarray, q) -> float:
  return float(np.percentile(x, q)) if x.size else float("nan")


def load_run(d: str):
  meta = json.load(open(os.path.join(d, "meta.json")))
  z = dict(np.load(os.path.join(d, "rollout.npz")))
  dec, P = meta["decimation"], meta["passive_steps"]
  T, n = z["invalid"].shape
  h = z["h"][dec - 1 :: dec]
  up = z["up"][dec - 1 :: dec].astype(np.float32)
  inv = z["invalid"] | z["nan_term"]
  fin = np.isfinite(z["qd"]).all(-1).reshape(T, dec, n).all(1)
  inv |= ~fin
  first_inv = np.where(inv.any(0), inv.argmax(0), T)
  last = int(round(LAST_S / meta["policy_dt"]))
  recovered = (h[-last:].mean(0) > H1) & (up[-last:].mean(0) > STAND_UP)
  strict = ((h[-last:] > H_SUCCESS) & (up[-last:] > STAND_UP)).all(0)
  cls = np.where(first_inv < T, "invalid", np.where(recovered, "success", "failed"))
  # Phases (policy-step resolution; first crossings).
  hh, uu = h[P:], up[P:]
  L = T - P
  w = int(round(STABLE_S / meta["policy_dt"]))
  ph = np.full((T, n), -1, np.int8)
  info = {k: np.full(n, np.nan) for k in ("t_h1", "t_h2", "t_stable")}
  regress_h1 = np.zeros(n, int)
  fell_after_stable = np.zeros(n, bool)
  for e in range(n):
    above1 = hh[:, e] > H1
    k1 = int(above1.argmax()) if above1.any() else L
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
    dt = meta["policy_dt"]
    if k1 < L:
      info["t_h1"][e] = k1 * dt
      # Down-crossings below h1 - 0.01 after the first crossing (regressions).
      x = hh[k1:, e]
      lo, st, cnt = H1 - 0.01, True, 0
      for v in x:
        if st and v < lo:
          st, cnt = False, cnt + 1
        elif not st and v > H1:
          st = True
      regress_h1[e] = cnt
    if k2 < L:
      info["t_h2"][e] = k2 * dt
    if ks < L:
      info["t_stable"][e] = ks * dt
      fell_after_stable[e] = bool((hh[ks:, e] < H1).any())
    if first_inv[e] < T:
      ph[first_inv[e] :, e] = -2  # excluded: explosion and after (reset)
  return {
    "meta": meta,
    "z": z,
    "h": h,
    "up": up,
    "cls": cls,
    "recovered": recovered,
    "strict": strict,
    "ph": ph,
    "ph_sub": np.repeat(ph, dec, axis=0),
    "first_inv": first_inv,
    "regress_h1": regress_h1,
    "fell_after_stable": fell_after_stable,
    **info,
  }


def joint_velocity_rows(r) -> list[dict]:
  z, meta = r["z"], r["meta"]
  it = meta["iteration"]
  aqd = np.abs(z["qd"])
  act = r["ph_sub"] >= 0  # actuated, valid part
  dtp = meta["physics_dt"]
  rows = []
  for cname in ("success", "failed", "invalid", "all_valid"):
    em = (r["cls"] != "invalid") if cname == "all_valid" else (r["cls"] == cname)
    if cname == "invalid":
      act_c = (r["ph_sub"] != -1) & em[None, :]  # actuated incl. post-explosion
      act_c &= np.repeat(
        np.arange(len(r["h"]))[:, None] <= r["first_inv"][None, :],
        meta["decimation"],
        0,
      )
    else:
      act_c = act & em[None, :]
    if not act_c.any():
      continue
    b_all = {}
    for thr in (6.28, SPIKE):
      b = (aqd > thr) & act_c[..., None]
      e0, j0, s0, s1 = _runs(np.transpose(b, (1, 2, 0)))
      b_all[thr] = (j0, (s1 - s0) * dtp * 1e3)
    for j, name in enumerate(SHORT):
      x = aqd[..., j][act_c]
      row = {
        "iteration": it,
        "episode_class": cname,
        "n_episodes": int(em.sum()),
        "joint": name,
        "samples": int(x.size),
        "mean_abs_qd": float(x.mean()),
        "p95_abs_qd": _pct(x, 95),
        "p99_abs_qd": _pct(x, 99),
        "max_abs_qd": float(x.max()),
      }
      for thr in THRESH:
        row[f"frac_above_{thr:g}"] = float((x > thr).mean())
      for thr, tag in ((6.28, "6p28"), (SPIKE, "20")):
        jj, dur = b_all[thr]
        dj = dur[jj == j]
        row[f"events_above_{tag}_per_episode"] = float(dj.size) / max(int(em.sum()), 1)
        row[f"event_{tag}_duration_ms_median"] = _pct(dj, 50)
        row[f"event_{tag}_duration_ms_p95"] = _pct(dj, 95)
        row[f"event_{tag}_duration_ms_max"] = float(dj.max()) if dj.size else 0.0
      rows.append(row)
  return rows


def _penetration(z, meta):
  qmin = np.array(meta["q_min"], np.float32)
  qmax = np.array(meta["q_max"], np.float32)
  q = z["q"]
  return np.maximum(np.maximum(qmin - q, q - qmax), 0.0)


def phase_rows(r) -> list[dict]:
  z, meta = r["z"], r["meta"]
  it, dt = meta["iteration"], meta["policy_dt"]
  aqd = np.abs(z["qd"])
  atau = np.abs(z["tau"].astype(np.float32))
  pen = _penetration(z, meta)
  names = meta["reward_terms"]
  rew = z["reward"]  # weighted per-second step values (stage r_w)
  qd_env_sub = -0.01 * np.square(np.maximum(aqd - 6.28, 0.0)).sum(-1)  # (S, n)
  dec = meta["decimation"]
  qd_env_sub_ps = qd_env_sub.reshape(-1, dec, qd_env_sub.shape[1]).mean(1)
  rows = []
  valid = r["cls"] != "invalid"
  for cname in ("all_valid", "success", "failed"):
    em = valid if cname == "all_valid" else (r["cls"] == cname)
    for p, pname in enumerate(PHASES):
      m = (r["ph"] == p) & em[None, :]
      ms = (r["ph_sub"] == p) & em[None, :]
      entered = m.any(0)
      if not entered.any():
        continue
      dur = m.sum(0)[entered] * dt
      nxt = (r["ph"] > p).any(0) & entered
      x = aqd[ms]
      t = atau[ms]
      pe = pen[ms]
      jp99 = [np.percentile(aqd[..., j][ms], 99) for j in range(len(SHORT))]
      top = np.argsort(jp99)[::-1][:3]
      row = {
        "iteration": it,
        "episode_class": cname,
        "phase": pname,
        "episodes_entering": int(entered.sum()),
        "completion_rate": float(nxt.sum() / entered.sum()) if p < 3 else float("nan"),
        "duration_s_mean": float(dur.mean()),
        "duration_s_median": float(np.median(dur)),
        "qd_p50": _pct(x, 50),
        "qd_p95": _pct(x, 95),
        "qd_p99": _pct(x, 99),
        "qd_max": float(x.max()),
        "frac_qd_above_6p28": float((x > 6.28).mean()),
        "frac_qd_above_10": float((x > 10).mean()),
        "frac_qd_above_20": float((x > 20).mean()),
        "tau_p95": _pct(t, 95),
        "tau_max": float(t.max()),
        "frac_tau_at_cap": float((t >= SAT).mean()),
        "limit_violation_frac": float((pe > PEN_EPS).mean()),
        "limit_violation_max_rad": float(pe.max()),
        "limit_violation_p99_rad_of_violating": _pct(pe[pe > PEN_EPS], 99),
        "top_joints_by_p99": ";".join(f"{SHORT[j]}:{jp99[j]:.1f}" for j in top),
        "share_of_all_qd_above_10_samples": float(
          (aqd[ms] > 10).sum()
          / max(((aqd > 10) & ((r["ph_sub"] >= 0) & em[None, :])[..., None]).sum(), 1)
        ),
      }
      row["rew_per_s_total"] = float(rew[m].sum(-1).mean())
      for k_, term in enumerate(names):
        row[f"rew_per_s_{term}"] = float(rew[..., k_][m].mean())
      row["rew_per_s_qd_soft_envelope_if_every_physics_step"] = float(
        qd_env_sub_ps[m].mean()
      )
      # Per-episode sums inside the phase (reward units = per-second value x dt).
      tot = (rew.sum(-1) * m).sum(0)[entered] * dt
      vel = sum(rew[..., names.index(t_)] for t_ in VEL_TERMS)
      row["episode_sum_total_in_phase"] = float(tot.mean())
      row["episode_sum_velocity_terms_in_phase"] = float(
        ((vel * m).sum(0)[entered] * dt).mean()
      )
      rows.append(row)
  return rows


def spike_events(r) -> list[dict]:
  z, meta = r["z"], r["meta"]
  it = meta["iteration"]
  dec, dtp = meta["decimation"], meta["physics_dt"]
  qd, q, tau = z["qd"], z["q"], z["tau"].astype(np.float32)
  tpd, qc = z["tau_pd"].astype(np.float32), z["qfrc_con"].astype(np.float32)
  qs, qcu = z["q_star"], z["q_cmd_unclipped"]
  qmin, qmax = np.array(meta["q_min"]), np.array(meta["q_max"])
  kp = np.array(meta["kp"])
  pen = _penetration(z, meta)
  S = qd.shape[0]
  act = r["ph_sub"] != -1  # actuated (incl. invalid tails, labelled)
  b = (np.abs(qd) > SPIKE) & act[..., None]
  e0, j0, s0, s1 = _runs(np.transpose(b, (1, 2, 0)))
  out = []
  for e, j, a, bnd in zip(e0, j0, s0, s1, strict=True):
    seg = np.abs(qd[a:bnd, e, j])
    pk = a + int(seg.argmax())
    sgn = np.sign(qd[pk, e, j])
    lo = max(pk - 200, 0)
    w = np.abs(qd[lo : pk + 1, e, j]) < 6.28
    rise = lo + (int(np.nonzero(w)[0][-1]) + 1 if w.any() else 0)
    win = slice(rise, pk + 1)
    i_act = float((sgn * tau[win, e, j]).sum() * dtp)
    i_con = float((sgn * qc[win, e, j]).sum() * dtp)
    sat_mot = float(
      ((np.abs(tau[win, e, j]) >= SAT) & (sgn * tau[win, e, j] > 0)).mean()
    )
    k_pk = pk // dec
    # Target steps of this joint in the 60 ms (3 policy steps) before the peak.
    ks = slice(max(k_pk - 2, 1), k_pk + 1)
    dqs = qs[ks, e, j] - qs[slice(ks.start - 1, ks.stop - 1), e, j]
    jump = float(np.abs(dqs).max()) if dqs.size else 0.0
    jump_dir = bool(dqs.size and (sgn * dqs).max() >= JUMP)
    jump_sat = jump >= TAU_CAP / kp[j]
    leg = LEG[0] if j in LEG[0] else LEG[1]
    dleg = np.abs(
      qs[ks, e][:, list(leg)] - qs[slice(ks.start - 1, ks.stop - 1), e][:, list(leg)]
    )
    leg_jump = float(dleg.max()) if dleg.size else 0.0
    w2 = slice(max(pk - 200, 0), min(pk + 201, S))
    pen_w = float(pen[w2, e, j].max())
    tch = z["touch"][w2, e]
    gained = np.bitwise_and(np.bitwise_not(tch[:-1]), tch[1:])
    lost = np.bitwise_and(tch[:-1], np.bitwise_not(tch[1:]))
    g_names = [
      SUPPORT_GROUPS[g]
      for g in range(len(SUPPORT_GROUPS))
      if (np.bitwise_or.reduce(gained) >> g) & 1
    ]
    T_ = len(r["h"])
    near_inv = r["first_inv"][e] < T_ and r["first_inv"][e] <= k_pk + 5
    pen_near = float(pen[max(pk - 40, 0) : pk + 41, e, j].max())
    # Priority: explosion > limit bounce > own target jump (direction of the peak) >
    # contact impulse > actuator saturation without a jump > coupling / unresolved.
    if near_inv:
      cause = 5
    elif pen_near > PEN_EPS and i_con > max(i_act, 0.0):
      cause = 4
    elif jump_dir:
      cause = 1
    elif i_con > 0 and i_con > i_act:
      cause = 3
    elif i_act > 0:
      cause = 2
    else:
      cause = 6
    ph = int(r["ph"][min(k_pk, len(r["ph"]) - 1), e])
    at_lim = (
      abs(qs[k_pk, e, j] - qmin[j]) < 1e-4 or abs(qs[k_pk, e, j] - qmax[j]) < 1e-4
    )
    out.append(
      {
        "iteration": it,
        "env": int(e),
        "pose": FALLEN_POSE_NAMES[int(z["pose"][e])],
        "episode_class": r["cls"][e],
        "joint": SHORT[j],
        "t_peak_s": round((pk + 1) * dtp, 4),
        "phase": PHASES[ph]
        if ph >= 0
        else ("passive" if ph == -1 else "after_invalid"),
        "qd_peak": round(float(qd[pk, e, j]), 3),
        "duration_above_20_ms": round((bnd - a) * dtp * 1e3, 2),
        "rise_from_6p28_ms": round((pk - rise) * dtp * 1e3, 2),
        "q": round(float(q[pk, e, j]), 4),
        "q_target": round(float(qs[k_pk, e, j]), 4),
        "q_cmd_unclipped": round(float(qcu[k_pk, e, j]), 4),
        "tracking_err": round(float(qs[k_pk, e, j] - q[pk, e, j]), 4),
        "tau": round(float(tau[pk, e, j]), 3),
        "tau_pd_unclipped": round(float(tpd[pk, e, j]), 2),
        "tau_at_cap_at_peak": bool(abs(tau[pk, e, j]) >= SAT),
        "sat_motoring_frac_rise": round(sat_mot, 3),
        "impulse_actuator_Nms": round(i_act, 5),
        "impulse_constraint_Nms": round(i_con, 5),
        "max_target_jump_60ms_rad": round(jump, 4),
        "target_jump_in_peak_direction_ge_0p25": jump_dir,
        "target_jump_alone_saturates_pd": bool(jump_sat),
        "max_same_leg_target_jump_60ms_rad": round(leg_jump, 4),
        "t_since_policy_start_s": round(
          (pk + 1) * dtp - meta["passive_steps"] * meta["policy_dt"], 4
        ),
        "target_at_limit": bool(at_lim),
        "limit_penetration_pm100ms_rad": round(pen_w, 4),
        "base_h": round(float(z["h"][pk, e]), 4),
        "base_up_cos": round(float(z["up"][pk, e]), 3),
        "contact_at_peak": "+".join(
          SUPPORT_GROUPS[g]
          for g in range(len(SUPPORT_GROUPS))
          if (int(z["touch"][pk, e]) >> g) & 1
        ),
        "contact_gained_pm100ms": "+".join(g_names),
        "contact_changes_pm100ms": int((gained != 0).sum() + (lost != 0).sum()),
        "valid": not near_inv,
        "cause": CAUSES[cause],
        "_pk": int(pk),
        "_j": int(j),
      }
    )
  return out


def target_rows(r) -> list[dict]:
  """Policy-target behaviour per joint and phase: 20 ms target steps, targets held at
  the joint-range clip, commands beyond the range."""
  z, meta = r["z"], r["meta"]
  qs, qcu = z["q_star"], z["q_cmd_unclipped"]
  qmin, qmax = np.array(meta["q_min"]), np.array(meta["q_max"])
  dq = np.abs(np.diff(qs, axis=0, prepend=qs[:1]))
  at_lim = (np.abs(qs - qmin) < 1e-4) | (np.abs(qs - qmax) < 1e-4)
  beyond = (qcu < qmin - 1e-4) | (qcu > qmax + 1e-4)
  valid = r["cls"] != "invalid"
  rows = []
  P = meta["passive_steps"]
  for p in range(4):
    m = (r["ph"] == p) & valid[None, :]
    m[P] = False  # the first policy step jumps from the passive hold (reported apart)
    for j, name in enumerate(SHORT):
      x = dq[..., j][m]
      if not x.size:
        continue
      rows.append(
        {
          "iteration": meta["iteration"],
          "phase": PHASES[p],
          "joint": name,
          "steps": int(x.size),
          "target_step_p50_rad": _pct(x, 50),
          "target_step_p95_rad": _pct(x, 95),
          "target_step_max_rad": float(x.max()),
          "frac_target_step_ge_0p25": float((x >= JUMP).mean()),
          "frac_target_at_range_limit": float(at_lim[..., j][m].mean()),
          "frac_command_beyond_range": float(beyond[..., j][m].mean()),
        }
      )
  first = dq[P][valid]
  for j, name in enumerate(SHORT):
    rows.append(
      {
        "iteration": meta["iteration"],
        "phase": "first_policy_step",
        "joint": name,
        "steps": int(first.shape[0]),
        "target_step_p50_rad": _pct(first[:, j], 50),
        "target_step_p95_rad": _pct(first[:, j], 95),
        "target_step_max_rad": float(first[:, j].max()),
        "frac_target_step_ge_0p25": float((first[:, j] >= JUMP).mean()),
        "frac_target_at_range_limit": float(at_lim[P][valid][:, j].mean()),
        "frac_command_beyond_range": float(beyond[P][valid][:, j].mean()),
      }
    )
  return rows


def effective_inertia(r) -> dict:
  """Joint-space inertia seen by each actuator, from saturated pushes starting near
  rest with no constraint force: I ~ tau / qddot (diagonal estimate, couplings
  ignored)."""
  z, meta = r["z"], r["meta"]
  dtp = meta["physics_dt"]
  tau = z["tau"].astype(np.float32)
  con = z["qfrc_con"].astype(np.float32)
  qd = z["qd"]
  qdd = (qd[1:] - qd[:-1]) / dtp
  act = (r["ph_sub"][1:] >= 0)[..., None]
  sel = (
    act
    & (np.abs(tau[1:]) >= TAU_CAP - 0.05)
    & (np.abs(con[1:]) < 0.2)
    & (np.abs(qd[:-1]) < 3.0)
  )
  sel &= np.sign(qdd) == np.sign(tau[1:])
  out = {}
  for j, name in enumerate(SHORT):
    x = (tau[1:, :, j] / qdd[..., j])[sel[..., j]]
    x = x[np.isfinite(x) & (x > 0)]
    out[name] = {
      "samples": int(x.size),
      "inertia_median_kgm2": _pct(x, 50),
      "inertia_p25_kgm2": _pct(x, 25),
      "inertia_p75_kgm2": _pct(x, 75),
      "time_to_20rad_s_at_16Nm_ms": 20.0 * _pct(x, 50) / TAU_CAP * 1e3
      if x.size
      else float("nan"),
    }
  return out


def limit_rows(r) -> list[dict]:
  z, meta = r["z"], r["meta"]
  qmin, qmax = np.array(meta["q_min"]), np.array(meta["q_max"])
  pen = _penetration(z, meta)
  aqd = np.abs(z["qd"])
  dec = meta["decimation"]
  qs = np.repeat(z["q_star"], dec, 0)
  qcu = np.repeat(z["q_cmd_unclipped"], dec, 0)
  valid = r["ph_sub"] >= 0
  rows = []
  for j, name in enumerate(SHORT):
    v = (pen[..., j] > PEN_EPS) & valid
    pv = pen[..., j][v]
    q = z["q"][..., j][v]
    side_lo = q < qmin[j]
    tgt = qs[..., j][v]
    tgt_at = np.where(
      side_lo, np.abs(tgt - qmin[j]) < 1e-4, np.abs(tgt - qmax[j]) < 1e-4
    )
    beyond = np.where(side_lo, qcu[..., j][v] < qmin[j], qcu[..., j][v] > qmax[j])
    inside = np.where(side_lo, tgt - qmin[j] > 0.05, qmax[j] - tgt > 0.05)
    phase_counts = {
      PHASES[p]: int(((r["ph_sub"] == p) & (pen[..., j] > PEN_EPS)).sum())
      for p in range(4)
    }
    rows.append(
      {
        "iteration": meta["iteration"],
        "joint": name,
        "q_min": float(qmin[j]),
        "q_max": float(qmax[j]),
        "violation_frac": float(v.sum() / max(valid.sum(), 1)),
        "episodes_with_violation": int(v.any(0).sum()),
        "max_rad": float(pv.max()) if pv.size else 0.0,
        "p99_rad_of_violating": _pct(pv, 99),
        "frac_lower_side": float(side_lo.mean()) if pv.size else float("nan"),
        "frac_with_qd_above_10": float((aqd[..., j][v] > 10).mean())
        if pv.size
        else float("nan"),
        "frac_target_clipped_at_that_limit": float(tgt_at.mean())
        if pv.size
        else float("nan"),
        "frac_policy_command_beyond_limit": float(beyond.mean())
        if pv.size
        else float("nan"),
        "frac_target_inside_by_0p05": float(inside.mean()) if pv.size else float("nan"),
        **{f"samples_{k}": c for k, c in phase_counts.items()},
      }
    )
  return rows


def reward_check(r) -> dict:
  """qd_soft_envelope as logged (50 Hz sample of the last physics step) vs the same
  formula at every physics step."""
  z, meta = r["z"], r["meta"]
  dec = meta["decimation"]
  aqd = np.abs(z["qd"])
  names = meta["reward_terms"]
  logged = z["reward"][..., names.index("qd_soft_envelope")]
  last = -0.01 * np.square(np.maximum(aqd[dec - 1 :: dec] - 6.28, 0)).sum(-1)
  full = -0.01 * np.square(np.maximum(aqd - 6.28, 0)).sum(-1).reshape(
    -1, dec, aqd.shape[1]
  ).mean(1)
  m = r["ph"] >= 0
  # Fraction of physics steps above 6.28 whose policy step's sample is <= 6.28
  # (missed by the 50 Hz sample), per joint pooled.
  over = (aqd > 6.28) & np.repeat(m, dec, 0)[..., None]
  samp = np.repeat(aqd[dec - 1 :: dec] > 6.28, dec, 0)
  missed = float((over & ~samp).sum() / max(over.sum(), 1))
  return {
    "iteration": meta["iteration"],
    "max_abs_diff_logged_vs_last_substep": float(np.abs(logged - last)[m].max()),
    "mean_logged_per_s": float(logged[m].mean()),
    "mean_if_every_physics_step_per_s": float(full[m].mean()),
    "frac_of_over_6p28_physics_samples_missed_by_50Hz_sample": missed,
  }


def analyze(args) -> None:
  out = args.out
  os.makedirs(out, exist_ok=True)
  jv, ph, sp, lim, chk, summ, tg, inert = [], [], [], [], [], [], [], {}
  primary = []
  for d in args.runs:
    r = load_run(d)
    it = r["meta"]["iteration"]
    run = os.path.basename(os.path.normpath(d))
    print(f"[analyze] {d} it {it}", flush=True)

    def tag(rows, run=run):
      return [{"run": run, **x} for x in rows]

    jv += tag(joint_velocity_rows(r))
    ph += tag(phase_rows(r))
    ev = tag(spike_events(r))
    sp += ev
    lim += tag(limit_rows(r))
    tg += tag(target_rows(r))
    inert[run] = effective_inertia(r)
    chk += tag([reward_check(r)])
    summ += tag([run_summary(r, ev)])
    summ[-1]["counterfactual_target_rate_limit"] = r["meta"].get(
      "counterfactual_target_rate_limit_rad_per_step"
    )
    if any(os.path.abspath(d) == os.path.abspath(p_) for p_ in args.primary):
      primary.append((it, select_videos(r, ev)))
      plots_primary(r, ev, out)
    del r
  _csv(os.path.join(out, "joint_velocity_statistics.csv"), jv)
  _csv(os.path.join(out, "phase_statistics.csv"), ph)
  _csv(
    os.path.join(out, "spike_events.csv"),
    [{k: v for k, v in e.items() if not k.startswith("_")} for e in sp],
  )
  _csv(os.path.join(out, "joint_limit_statistics.csv"), lim)
  _csv(os.path.join(out, "target_statistics.csv"), tg)
  with open(os.path.join(out, "analysis.json"), "w") as f:
    json.dump(
      {"runs": summ, "reward_check": chk, "effective_inertia": inert}, f, indent=1
    )
  base = [x for x in summ if not x["run"].startswith("counterfactual")]
  keep = {x["run"] for x in base}
  plots_compare(
    base,
    [x for x in ph if x["run"] in keep],
    [x for x in jv if x["run"] in keep],
    [x for x in sp if x["run"] in keep],
    out,
  )
  with open(os.path.join(out, "video_selection.json"), "w") as f:
    json.dump({str(it): s_ for it, s_ in primary}, f, indent=1)
  print("[analyze] done", flush=True)


def _csv(path, rows) -> None:
  keys = list(rows[0].keys())
  for r_ in rows:
    for k in r_:
      if k not in keys:
        keys.append(k)
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r_ in rows:
      w.writerow(
        {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r_.items()}
      )


def run_summary(r, ev) -> dict:
  meta, cls = r["meta"], r["cls"]
  poses = np.array(r["z"]["pose"])
  causes = {}
  for e in ev:
    causes[e["cause"]] = causes.get(e["cause"], 0) + 1
  aqd = np.abs(r["z"]["qd"])
  valid_act = (r["ph_sub"] >= 0)[..., None]
  return {
    "iteration": meta["iteration"],
    "checkpoint": meta["checkpoint"],
    "episodes": int(len(cls)),
    "success": int((cls == "success").sum()),
    "failed": int((cls == "failed").sum()),
    "invalid": int((cls == "invalid").sum()),
    "recovered_by_pose": {
      FALLEN_POSE_NAMES[p]: float(r["recovered"][poses == p].mean()) for p in range(4)
    },
    "strict_by_pose": {
      FALLEN_POSE_NAMES[p]: float(r["strict"][poses == p].mean()) for p in range(4)
    },
    "t_h1_median_s": float(np.nanmedian(r["t_h1"])),
    "t_h2_median_s": float(np.nanmedian(r["t_h2"])),
    "t_stable_median_s": float(np.nanmedian(r["t_stable"])),
    "regressions_below_h1_mean": float(r["regress_h1"].mean()),
    "fell_after_stable": int(r["fell_after_stable"].sum()),
    "qd_max_valid": float(aqd[np.broadcast_to(valid_act, aqd.shape)].max()),
    "spike_events_above_20": len(ev),
    "spike_causes": causes,
  }


def select_videos(r, ev) -> dict:
  cls = r["cls"]
  aqd = np.abs(r["z"]["qd"])
  valid = r["ph_sub"] >= 0
  peak = np.where(valid[..., None], aqd, 0).max(axis=(0, 2))
  succ = np.nonzero(cls == "success")[0]
  sel = {}
  if succ.size:
    sel["success_high_qd"] = int(succ[np.argmax(peak[succ])])
    sel["success_low_qd"] = int(succ[np.argmin(peak[succ])])
  fail = np.nonzero(cls == "failed")[0]
  if fail.size:
    sel["failed"] = int(fail[np.argmax(peak[fail])])
  inv = np.nonzero(cls == "invalid")[0]
  if inv.size:
    sel["physics_explosion"] = int(inv[0])
  return {
    k: {"env": v, "peak_qd": float(peak[v]), "class": str(cls[v])}
    for k, v in sel.items()
  }


# ---------------------------------------------------------------------------
# Plots (matplotlib, light surface; 3 highlighted series + gray context)

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, MUTED = "#b4b2aa", "#0b0b0b", "#52514e"


def _style():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {
      "figure.facecolor": "#fcfcfb",
      "axes.facecolor": "#fcfcfb",
      "axes.edgecolor": GRAY,
      "axes.labelcolor": MUTED,
      "xtick.color": MUTED,
      "ytick.color": MUTED,
      "axes.grid": True,
      "grid.color": "#e6e5e0",
      "grid.linewidth": 0.6,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "font.size": 9,
      "axes.titlesize": 10,
      "axes.titlecolor": INK,
      "legend.frameon": False,
    }
  )
  return plt


def _marks(ax, r, e):
  P = r["meta"]["passive_steps"] * r["meta"]["policy_dt"]
  for key in ("t_h1", "t_h2", "t_stable"):
    v = r[key][e]
    if np.isfinite(v):
      ax.axvline(P + v + r["meta"]["policy_dt"], color=MUTED, lw=0.8, ls=":")


def trace_plot(r, e, path, title, window=None):
  plt = _style()
  z, meta = r["z"], r["meta"]
  dtp = meta["physics_dt"]
  S = z["qd"].shape[0]
  t = (np.arange(S) + 1) * dtp
  aq = np.abs(z["qd"][:, e])
  valid = r["ph_sub"][:, e] >= 0
  pk = np.where(valid[:, None], aq, 0).max(0)
  top = np.argsort(pk)[::-1][:3]
  j = int(top[0])
  a, b = (
    (0, S) if window is None else (int(window[0] / dtp), min(int(window[1] / dtp), S))
  )
  sl = slice(a, b)
  fig, ax = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)
  for jj in range(aq.shape[1]):
    if jj not in top:
      ax[0].plot(t[sl], aq[sl, jj], color=GRAY, lw=0.6)
  ax[0].plot([], [], color=GRAY, lw=0.6, label="other 9 joints")
  for c, jj in zip((C1, C2, C3), top, strict=False):
    ax[0].plot(
      t[sl], aq[sl, jj], color=c, lw=1.0, label=f"{SHORT[jj]} (peak {pk[jj]:.1f})"
    )
  for thr in (6.28, 20.0):
    ax[0].axhline(thr, color=MUTED, lw=0.7, ls="--")
  kk = a + int(aq[sl, j].argmax())
  ax[0].plot(t[kk], aq[kk, j], "o", ms=8, mfc="none", mec=INK)
  ax[0].set_ylabel("|qdot| (rad/s), 0.5 ms")
  ax[0].legend(loc="upper right", fontsize=8)
  ax[0].set_title(title)
  qs = np.repeat(z["q_star"][:, e, j], meta["decimation"])
  ax[1].plot(t[sl], z["q"][sl, e, j], color=C1, lw=1.2, label="q (measured)")
  ax[1].plot(
    t[sl], qs[sl], color=C2, lw=1.0, drawstyle="steps-post", label="q* (target, 50 Hz)"
  )
  ax[1].axhline(meta["q_min"][j], color=MUTED, lw=0.8, ls="--")
  ax[1].axhline(meta["q_max"][j], color=MUTED, lw=0.8, ls="--", label="joint range")
  qj = z["q"][sl, e, j]
  pen = np.maximum(np.maximum(meta["q_min"][j] - qj, qj - meta["q_max"][j]), 0)
  vi = np.nonzero(pen > PEN_EPS)[0]
  if vi.size:
    ax[1].plot(
      t[sl][vi], qj[vi], "|", color=INK, ms=6, label="limit violation > 5 mrad"
    )
  ax[1].set_ylabel(f"{SHORT[j]} (rad)")
  ax[1].legend(loc="upper right", fontsize=8)
  ax[2].plot(
    t[sl],
    z["tau_pd"][sl, e, j].astype(np.float32).clip(-60, 60),
    color=C2,
    lw=0.8,
    label="PD before clip (shown within +-60)",
  )
  ax[2].plot(
    t[sl],
    z["tau"][sl, e, j].astype(np.float32),
    color=C1,
    lw=1.1,
    label="applied torque",
  )
  ax[2].plot(
    t[sl],
    z["qfrc_con"][sl, e, j].astype(np.float32).clip(-60, 60),
    color=C3,
    lw=0.8,
    label="joint constraint force (contact + limit)",
  )
  for v in (-TAU_CAP, TAU_CAP):
    ax[2].axhline(v, color=MUTED, lw=0.7, ls="--")
  ax[2].set_ylabel(f"{SHORT[j]} (Nm)")
  ax[2].legend(loc="upper right", fontsize=8)
  ax[3].plot(t[sl], z["h"][sl, e], color=C1, lw=1.2, label="base height")
  for v, lab in ((H1, "h1 0.19"), (H2, "h2 0.276")):
    ax[3].axhline(v, color=MUTED, lw=0.7, ls="--")
    ax[3].text(t[sl][0], v + 0.005, lab, color=MUTED, fontsize=8)
  tch = z["touch"][sl, e]
  on = np.nonzero(np.bitwise_and(np.bitwise_not(tch[:-1]), tch[1:]))[0]
  if on.size:
    ax[3].plot(
      t[sl][on + 1],
      np.full(on.size, 0.02),
      "|",
      color=C2,
      ms=8,
      label="contact onset (any body group)",
    )
  ax[3].set_ylabel("m")
  ax[3].set_xlabel(
    "time since episode start (s); dotted: first h1, first h2, stable (0.5 s > h2, upright)"
  )
  ax[3].legend(loc="lower right", fontsize=8)
  for x in ax:
    _marks(x, r, e)
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def event_plot(r, evs, path):
  plt = _style()
  z, meta = r["z"], r["meta"]
  dtp, dec = meta["physics_dt"], meta["decimation"]
  fig, axes = plt.subplots(len(evs), 3, figsize=(13, 3.0 * len(evs)), squeeze=False)
  for row, ev in zip(axes, evs, strict=True):
    e, j, pk = ev["env"], ev["_j"], ev["_pk"]
    a, b = max(pk - 200, 0), min(pk + 201, z["qd"].shape[0])
    t = (np.arange(a, b) - pk) * dtp * 1e3
    row[0].plot(t, z["qd"][a:b, e, j], color=C1, lw=1.2)
    row[0].axhline(0, color=GRAY, lw=0.6)
    row[0].set_title(
      f"it {ev['iteration']} env {e} {ev['joint']} t={ev['t_peak_s']:.3f}s | {ev['cause'][:40]}",
      fontsize=8.5,
    )
    row[0].set_ylabel("qdot (rad/s)")
    qs = np.repeat(z["q_star"][:, e, j], dec)[a:b]
    row[1].plot(t, z["q"][a:b, e, j], color=C1, lw=1.2, label="q")
    row[1].plot(t, qs, color=C2, lw=1.0, drawstyle="steps-post", label="q* target")
    row[1].axhline(meta["q_min"][j], color=MUTED, lw=0.7, ls="--")
    row[1].axhline(meta["q_max"][j], color=MUTED, lw=0.7, ls="--")
    row[1].legend(fontsize=7, loc="best")
    row[1].set_ylabel("rad")
    row[2].plot(
      t, z["tau"][a:b, e, j].astype(np.float32), color=C1, lw=1.2, label="applied"
    )
    row[2].plot(
      t,
      z["tau_pd"][a:b, e, j].astype(np.float32).clip(-60, 60),
      color=C2,
      lw=0.8,
      label="PD unclipped",
    )
    row[2].plot(
      t,
      z["qfrc_con"][a:b, e, j].astype(np.float32).clip(-60, 60),
      color=C3,
      lw=0.9,
      label="constraint",
    )
    row[2].legend(fontsize=7, loc="best")
    row[2].set_ylabel("Nm")
    for x in row:
      x.axvline(0, color=MUTED, lw=0.7, ls=":")
      x.set_xlabel("ms from peak")
  fig.tight_layout()
  fig.savefig(path, dpi=105)
  plt.close(fig)


def dist_plot(r, path):
  plt = _style()
  aqd = np.abs(r["z"]["qd"])
  valid = r["cls"] != "invalid"
  p99 = np.full((len(SHORT), 4), np.nan)
  f10 = np.full((len(SHORT), 4), np.nan)
  for p in range(4):
    ms = (r["ph_sub"] == p) & valid[None, :]
    for j in range(len(SHORT)):
      x = aqd[..., j][ms]
      if x.size:
        p99[j, p] = np.percentile(x, 99)
        f10[j, p] = (x > 10).mean() * 100
  fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
  for a_, m, lab in (
    (ax[0], p99, "p99 |qdot| (rad/s)"),
    (ax[1], f10, "% of physics steps with |qdot| > 10 rad/s"),
  ):
    im = a_.imshow(m, cmap="Blues", aspect="auto")
    a_.set_xticks(range(4), PHASES, rotation=20)
    a_.set_yticks(range(len(SHORT)), SHORT)
    a_.grid(False)
    a_.set_title(lab)
    for (y, x), v in np.ndenumerate(m):
      if np.isfinite(v):
        a_.text(
          x,
          y,
          f"{v:.1f}",
          ha="center",
          va="center",
          fontsize=7.5,
          color="white" if v > np.nanmax(m) * 0.6 else INK,
        )
    fig.colorbar(im, ax=a_, shrink=0.8)
  fig.suptitle(
    f"it {r['meta']['iteration']}: valid episodes, by joint and recovery phase",
    color=INK,
  )
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def plots_primary(r, ev, out) -> None:
  pdir = os.path.join(out, "plots")
  os.makedirs(pdir, exist_ok=True)
  it = r["meta"]["iteration"]
  sel = select_videos(r, ev)
  P = r["meta"]["passive_steps"] * r["meta"]["policy_dt"]
  for tag, s_ in sel.items():
    e = s_["env"]
    pose = FALLEN_POSE_NAMES[int(r["z"]["pose"][e])]
    title = f"it {it} env {e} ({pose}, {s_['class']}): {tag}"
    trace_plot(r, e, os.path.join(pdir, f"trace_it{it}_{tag}_full.png"), title)
    end = r["t_stable"][e]
    end = P + (end if np.isfinite(end) else 3.0) + 1.0
    trace_plot(
      r,
      e,
      os.path.join(pdir, f"trace_it{it}_{tag}_zoom.png"),
      title + " (zoom)",
      (P - 0.1, end),
    )
  dist_plot(r, os.path.join(pdir, f"qd_by_joint_phase_it{it}.png"))
  by = {}
  for e_ in sorted(ev, key=lambda x: -abs(x["qd_peak"])):
    by.setdefault(e_["cause"], []).append(e_)
  pick = [v[0] for v in by.values()][:4]
  if by:
    common = max(by.values(), key=len)
    pick.append(common[len(common) // 2])
  if pick:
    event_plot(r, pick, os.path.join(pdir, f"spike_events_it{it}.png"))


def plots_compare(summ, ph, jv, sp, out) -> None:
  plt = _style()
  pdir = os.path.join(out, "plots")
  os.makedirs(pdir, exist_ok=True)
  its = sorted({s_["iteration"] for s_ in summ})
  fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
  for c, pname in zip((C1, C2, C3, C4), PHASES, strict=True):
    for a_, key in ((ax[0], "qd_p95"), (ax[1], "qd_p99")):
      ys = [
        next(
          (
            x[key]
            for x in ph
            if x["iteration"] == i
            and x["phase"] == pname
            and x["episode_class"] == "all_valid"
          ),
          np.nan,
        )
        for i in its
      ]
      a_.plot(its, ys, "-o", color=c, lw=2, ms=5, label=pname)
  ax[0].set_title("p95 |qdot| by phase (rad/s)")
  ax[1].set_title("p99 |qdot| by phase (rad/s)")
  ax[0].legend(fontsize=8)
  rec = [
    np.mean(list(s_["recovered_by_pose"].values())) * 100
    for s_ in sorted(summ, key=lambda x: x["iteration"])
  ]
  ax[2].plot(its, rec, "-o", color=C1, lw=2, ms=5)
  ax[2].set_ylim(0, 105)
  ax[2].set_title("no-assist recovery, 32 envs/pose (%)")
  for a_ in ax:
    a_.set_xlabel("iteration")
  fig.tight_layout()
  fig.savefig(os.path.join(pdir, "compare_iterations.png"), dpi=110)
  plt.close(fig)
  fig, ax = plt.subplots(figsize=(10, 4.2))
  bottom = np.zeros(len(its))
  cols = (C1, C2, C3, C4, "#e87ba4", GRAY)
  for c, cause in zip(cols, CAUSES.values(), strict=True):
    v = np.array(
      [sum(1 for e in sp if e["iteration"] == i and e["cause"] == cause) for i in its],
      float,
    )
    if v.sum() == 0:
      continue
    ax.bar(
      [str(i) for i in its],
      v,
      bottom=bottom,
      color=c,
      label=cause,
      edgecolor="#fcfcfb",
      linewidth=2,
    )
    bottom += v
  ax.set_title("|qdot| > 20 rad/s events by cause (128 episodes per checkpoint)")
  ax.set_xlabel("iteration")
  ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
  fig.tight_layout()
  fig.savefig(os.path.join(pdir, "spike_causes_by_iteration.png"), dpi=110)
  plt.close(fig)


def videos(args) -> None:
  import imageio.v2 as imageio

  from minipi_getup.ftsr_ref.analyze_recovery import _scene_model
  from minipi_getup.ftsr_ref.visual_demo import Renderer

  r = load_run(args.run)
  meta, z = r["meta"], r["z"]
  sel = json.load(open(os.path.join(args.out, "video_selection.json")))[
    str(meta["iteration"])
  ]
  rend = Renderer(_scene_model())
  vdir = os.path.join(args.out, "videos")
  os.makedirs(vdir, exist_ok=True)
  dec, dt = meta["decimation"], meta["policy_dt"]
  origins = np.array(meta["origins"])
  aqd = np.abs(z["qd"])
  it = meta["iteration"]
  P = meta["passive_steps"]
  rec = {}
  for tag, s_ in sel.items():
    e = s_["env"]
    pose = FALLEN_POSE_NAMES[int(z["pose"][e])]
    T = z["qpos"].shape[0]
    stepmax = aqd[:, e].reshape(T, dec, -1).max(1)  # (T, nj) max over the step
    atcap = (
      (np.abs(z["tau"][:, e].astype(np.float32)) >= SAT)
      .reshape(T, dec, -1)
      .any(1)
      .sum(-1)
    )
    frames = []
    rend.lookat = None
    for k in range(T):
      img = rend.frame(z["qpos"][k, e], origins[e])
      j = int(stepmax[k].argmax())
      ph = int(r["ph"][k, e])
      phase = (
        "passive (kp 0, kd 1)" if ph == -1 else ("invalid" if ph == -2 else PHASES[ph])
      )
      lines = [
        f"it {it} env {e} {pose}  [{s_['class']}]  {tag}",
        f"t = {(k + 1) * dt:5.2f} s   phase {phase}",
        f"base height {r['h'][k, e]:.3f} m  (h1 0.19, h2 0.276)",
        f"max |qdot| in step {stepmax[k, j]:5.1f} rad/s ({SHORT[j]})",
        f"joints at torque cap in step: {int(atcap[k])}",
        "recorded trajectory, zero assistance, deterministic",
      ]
      frames.append(rend.overlay(img, lines))
    path = os.path.join(vdir, f"it{it}_{tag}_env{e}_realtime.mp4")
    imageio.mimwrite(path, frames, fps=int(round(1.0 / dt)), quality=8)
    # Slow motion (0.2x) of the actuated part up to stable + 1 s.
    ts = r["t_stable"][e]
    k_end = min(T, P + int(((ts if np.isfinite(ts) else 4.0) + 1.0) / dt))
    slow = os.path.join(vdir, f"it{it}_{tag}_env{e}_slow0p2x.mp4")
    imageio.mimwrite(slow, frames[P - 5 : k_end], fps=10, quality=8)
    rec[tag] = {"env": e, "pose": pose, "realtime": path, "slow_0p2x": slow, **s_}
  with open(os.path.join(vdir, "videos.json"), "w") as f:
    json.dump(rec, f, indent=1)
  print("[videos] done", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  sp = ap.add_subparsers(dest="cmd", required=True)
  r = sp.add_parser("rollout")
  r.add_argument("--checkpoint", required=True)
  r.add_argument("--out", required=True)
  r.add_argument("--per-pose", type=int, default=32)
  r.add_argument("--seed", type=int, default=2150)
  r.add_argument(
    "--target-rate-limit",
    type=float,
    default=0.0,
    help="COUNTERFACTUAL eval only: max PD-target change per policy step (rad)",
  )
  a = sp.add_parser("analyze")
  a.add_argument("--runs", nargs="+", required=True)
  a.add_argument("--out", required=True)
  a.add_argument("--primary", nargs="+", required=True)
  v = sp.add_parser("videos")
  v.add_argument("--run", required=True)
  v.add_argument("--out", required=True)
  args = ap.parse_args()
  {"rollout": rollout, "analyze": analyze, "videos": videos}[args.cmd](args)


if __name__ == "__main__":
  main()
