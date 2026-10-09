"""Phase A static validation (rebuild prompt Sec. 51) plus the action-contract checks.

Run:  WARP_CACHE_PATH=... uv run python -m minipi_getup.ftsr_ref.validate [--only 7,21]

Every test prints PASS / FAIL with its measured numbers. Simulation only.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import tempfile
import time
import traceback

import numpy as np
import torch
import yaml

import minipi_getup  # noqa: F401  (registers the tasks)
from minipi_getup.ftsr_ref import mdp
from minipi_getup.ftsr_ref.config.env_cfg import (
  ACTION_SCALE,
  ASSIST_F_MAX,
  PASSIVE_STEPS,
  RAW_CLIP,
  STAGE_HEIGHTS,
  STAGE_REWARD_HEIGHTS,
)
from minipi_getup.ftsr_ref.config.robot import (
  JOINT_NAMES,
  JOINT_RANGES,
  KD,
  KP,
  MINIPI_MASS,
  MINIPI_WEIGHT,
  TAU_CAP,
)
from minipi_getup.ftsr_ref.mdp.actions import COST_KEY
from minipi_getup.ftsr_ref.mdp.assistance import time_coeff, uprighting_rotvec
from minipi_getup.ftsr_ref.mdp.stages import StageCfg, decide_stage
from minipi_getup.ftsr_ref.rl.storage import (
  RolloutStorage,
  discounted_cost_to_go,
  gae,
  mixed_advantage,
)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
WALK = "Mjlab-FTSR-Ref-MiniPi-Walk"
REC = "Mjlab-FTSR-Ref-MiniPi-Recovery"
REC_STATELESS = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless"
OLD_OMEGA0 = 75.0 * 2.0 * math.pi / 60.0  # removed H-conservative no-load speed
DEPLOY_YAML = (
  "/home/huy/Hightorque_Pi/mini_pi_fsm/deploy/robots/mini_pi/config/policy/velocity/"
  "mjlab47/params/deploy.yaml"
)
RESULTS: list[tuple[int, str, bool, str]] = []


def make_env(task, n, play=False, mutate=None):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = copy.deepcopy(load_env_cfg(task, play=play))
  cfg.scene.num_envs = n
  if mutate is not None:
    mutate(cfg)
  env = ManagerBasedRlEnv(cfg, device=DEV)
  env.reset()
  return env


def term(env):
  return env.action_manager.get_term("joint_pos")


def per_joint(table):
  out = []
  for name in JOINT_NAMES:
    hits = [v for k, v in table.items() if re.fullmatch(k, name)]
    out.append(hits[0])
  return np.array(out)


def record(num: int, name: str, ok: bool, info: str) -> None:
  RESULTS.append((num, name, ok, info))
  print(f"[{'PASS' if ok else 'FAIL'}] {num:2d} {name}: {info}", flush=True)


# ---------------------------------------------------------------------------------


def t01_joint_order(_):
  with open(DEPLOY_YAML) as f:
    deploy = yaml.safe_load(f)
  env = make_env(WALK, 2)
  names = term(env).joint_names
  ok = tuple(names) == JOINT_NAMES == tuple(deploy["joint_names"])
  env.close()
  return ok, f"policy order {names[0]}..{names[-1]} == mjlab47 deploy.yaml joint_names"


def t02_signs_default(_):
  with open(DEPLOY_YAML) as f:
    deploy = yaml.safe_load(f)
  env = make_env(WALK, 2)
  m = env.sim.mj_model
  robot = env.scene["robot"]
  axes = {}
  for name in JOINT_NAMES:
    jid = m.joint("robot/" + name).id
    axes[name] = tuple(np.round(m.jnt_axis[jid], 3))
  expect = {
    "hip_pitch": (0, 1, 0),
    "hip_roll": (1, 0, 0),
    "thigh": (0, 0, 1),
    "calf": (0, 1, 0),
    "ankle_pitch": (0, 1, 0),
    "ankle_roll": (1, 0, 0),
  }
  ok_axes = all(axes[n] == expect[n[2:].replace("_joint", "")] for n in JOINT_NAMES)
  default = robot.data.default_joint_pos[0].cpu().numpy()
  ok_def = np.allclose(default, 0.0) and np.allclose(deploy["default_joint_pos"], 0.0)
  env.close()
  return ok_axes and ok_def, (
    f"axes match (pitch y, roll x, yaw z, both legs same direction): {ok_axes}; "
    f"default pose all 0 in sim and deploy: {ok_def}"
  )


def t03_joint_limits(_):
  env = make_env(WALK, 4)
  m = env.sim.mj_model
  rng = np.array([m.jnt_range[m.joint("robot/" + n).id] for n in JOINT_NAMES])
  ref = np.array([JOINT_RANGES[n] for n in JOINT_NAMES])
  t = term(env)
  ok_model = np.allclose(rng, ref)
  ok_term = np.allclose(t.q_min.cpu(), ref[:, 0]) and np.allclose(
    t.q_max.cpu(), ref[:, 1]
  )
  # Extreme actions for many steps: the commanded target never leaves the ranges.
  worst = 0.0
  for sign in (1.0, -1.0):
    for _ in range(60):
      env.step(torch.full((4, 12), sign * 1e3, device=DEV))
      q = t.q_star.cpu().numpy()
      worst = max(worst, float(np.max(ref[:, 0] - q)), float(np.max(q - ref[:, 1])))
  env.close()
  ok = ok_model and ok_term and worst <= 1e-6
  return (
    ok,
    f"XML = vendor ranges {ok_model}, clip bounds {ok_term}, max excess {worst:.2e}",
  )


def t04_pd_gains(_):
  with open(DEPLOY_YAML) as f:
    deploy = yaml.safe_load(f)
  env = make_env(WALK, 2)
  t = term(env)
  model = env.sim.model
  kp = model.actuator_gainprm[0, t._ctrl, 0].cpu().numpy()
  kd = -model.actuator_biasprm[0, t._ctrl, 2].cpu().numpy()
  bias_p = -model.actuator_biasprm[0, t._ctrl, 1].cpu().numpy()
  ref_kp = np.array([KP[n[2:].replace("_joint", "")] for n in JOINT_NAMES])
  ref_kd = np.array([KD[n[2:].replace("_joint", "")] for n in JOINT_NAMES])
  ok = (
    np.allclose(kp, ref_kp)
    and np.allclose(bias_p, ref_kp)
    and np.allclose(kd, ref_kd)
    and np.allclose(ref_kp, deploy["stiffness"])
    and np.allclose(ref_kd, deploy["damping"])
  )
  arm = env.sim.mj_model.dof_armature[6:]
  env.close()
  return ok and np.allclose(arm, 0.0), (
    f"kp {kp[:6].tolist()} kd {kd[:6].tolist()} = deploy.yaml; armature max "
    f"{arm.max():.3g} (none added)"
  )


def t05_timing(_):
  env = make_env(WALK, 2)
  dt = env.sim.mj_model.opt.timestep
  ok = (
    abs(dt - 0.0005) < 1e-12
    and env.cfg.decimation == 40
    and abs(env.step_dt - 0.02) < 1e-12
  )
  info = f"physics {dt} s x decimation {env.cfg.decimation} = policy {env.step_dt} s"
  env.close()
  return ok, info


def t06_target_hold(_):
  env = make_env(WALK, 4)
  t = term(env)
  ctrl_log: list[torch.Tensor] = []
  step = env.sim.step

  def logged():
    ctrl_log.append(env.sim.data.ctrl[:, t._ctrl].clone())
    step()

  env.sim.step = logged
  worst = 0.0
  for _ in range(5):
    ctrl_log.clear()
    env.step(torch.randn(4, 12, device=DEV))
    c = torch.stack(ctrl_log)
    worst = max(worst, float((c - c[0]).abs().max()))
    worst = max(worst, float((c[0] - t.q_star).abs().max()))
  env.sim.step = step
  env.close()
  return (
    worst == 0.0,
    f"{len(ctrl_log)} physics steps per policy step, ctrl spread {worst}",
  )


def _reference_pipeline(a, default, scale, lo, hi):
  """raw clip -> scale -> physical clip; no slew (pd16_noslew)."""
  a_c = np.clip(a, -RAW_CLIP, RAW_CLIP)
  return np.clip(default + scale * a_c, lo, hi), a_c


def t07_action_mapping(_):
  env = make_env(WALK, 8)
  t = term(env)
  scale = per_joint(ACTION_SCALE)
  ok_scale = np.allclose(t.scale.cpu().numpy(), scale)
  lo = np.array([JOINT_RANGES[n][0] for n in JOINT_NAMES])
  hi = np.array([JOINT_RANGES[n][1] for n in JOINT_NAMES])
  gen = torch.Generator(device="cpu").manual_seed(0)
  worst, max_jump, in_range = 0.0, 0.0, True
  prev = None
  for k in range(40):
    mag = [0.05, 1.0, 5.0, 80.0][k % 4]
    a = (torch.randn(8, 12, generator=gen) * mag).numpy()
    ref, _ = _reference_pipeline(a, 0.0, scale, lo, hi)
    env.step(torch.tensor(a, device=DEV, dtype=torch.float))
    got = t.q_star.cpu().numpy()
    worst = max(worst, float(np.abs(got - ref).max()))
    in_range &= bool((got >= lo - 1e-6).all() and (got <= hi + 1e-6).all())
    if prev is not None:
      max_jump = max(max_jump, float(np.abs(got - prev).max()))
    prev = got
  env.close()
  # No slew: consecutive targets jump far beyond the old 0.06 rad / step.
  ok = ok_scale and worst < 1e-5 and in_range and max_jump > 1.0
  return ok, (
    f"per-joint scale {scale[:6].tolist()}; |sim - (raw clip -> scale -> range clip)| "
    f"max {worst:.1e}; targets inside XML ranges {in_range}; no slew: max "
    f"|q*_t - q*_t-1| {max_jump:.2f} rad (old limit 0.06)"
  )


def t08_action_clip(_):
  env = make_env(WALK, 4, play=True)
  t = term(env)
  a = torch.tensor([[1e3, -1e3] * 6] * 4, device=DEV)
  env.step(a)
  ok_raw = torch.allclose(t.raw_action, torch.clamp(a, -RAW_CLIP, RAW_CLIP))
  obs = env.get_observations()
  ok_obs = torch.allclose(obs["actor"][:, 36:48], t.raw_action)
  env.close()
  return (
    ok_raw and ok_obs,
    f"raw clip +-{RAW_CLIP}: action term {ok_raw}, last_action obs {ok_obs}",
  )


def _floating_joint_run(jn: str, frac: float = 0.9):
  """Robot floating (no gravity, no contacts); square-wave targets on one joint.
  Returns per physics step (qd, force, unclipped PD) of that joint."""

  def mutate(cfg):
    cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
    cfg.sim.mujoco.disableflags = ("contact",)
    cfg.terminations.pop("fell")

  env = make_env(WALK, 1, play=True, mutate=mutate)
  t = term(env)
  j = JOINT_NAMES.index(jn)
  lo, hi = JOINT_RANGES[jn]
  kp = KP[jn[2:].replace("_joint", "")]
  kd = KD[jn[2:].replace("_joint", "")]
  rows = []
  step = env.sim.step

  def logged():
    d = env.scene["robot"].data
    q = float(d.joint_pos[0, t._ids[j]])
    qd = float(d.joint_vel[0, t._ids[j]])
    pd = kp * (float(t.q_star[0, j]) - q) - kd * qd
    step()
    rows.append((qd, float(env.sim.data.actuator_force[0, t._ctrl[j]]), pd))

  env.sim.step = logged
  for k in range(60):
    a = torch.zeros(1, 12, device=DEV)
    a[:, j] = (frac * hi if (k // 15) % 2 == 0 else frac * lo) / float(t.scale[j])
    env.step(a)
  env.sim.step = step
  env.close()
  return np.array(rows)


def t09_no_derating(_):
  """The cap is the only limit: where the unclipped PD exceeds 16 Nm, the force is
  +-16 Nm at any speed, including speeds above the removed 7.85 rad/s no-load speed
  (the old envelope gave 0 Nm motoring torque there)."""
  out, ok = [], True
  for jn in ("r_hip_pitch_joint", "r_calf_joint"):
    r = _floating_joint_run(jn)
    qd, f, pd = r[:, 0], r[:, 1], r[:, 2]
    sat = np.abs(pd) > TAU_CAP
    motoring_fast = sat & (np.sign(pd) == np.sign(qd)) & (np.abs(qd) > OLD_OMEGA0)
    err_sat = float(np.abs(np.abs(f[sat]) - TAU_CAP).max()) if sat.any() else 0.0
    ok &= bool(motoring_fast.any()) and err_sat < 1e-3
    out.append(
      f"{jn[2:-6]}: {int(motoring_fast.sum())} saturated motoring steps at "
      f"|qd| > {OLD_OMEGA0:.2f} rad/s (max {np.abs(qd[motoring_fast]).max() if motoring_fast.any() else 0:.1f}), "
      f"|force| there = 16 Nm (err {err_sat:.1e}); peak |qd| {np.abs(qd).max():.1f}"
    )
  return ok, "; ".join(out)


def _substep_check(task, steps=150, n=32):
  env = make_env(task, n)
  t = term(env)
  step = env.sim.step
  stats = {"absmax": 0.0, "err": 0.0, "qdmax": 0.0, "sat": 0}
  kp = torch.tensor([KP[n[2:].replace("_joint", "")] for n in JOINT_NAMES], device=DEV)
  kd = torch.tensor([KD[n[2:].replace("_joint", "")] for n in JOINT_NAMES], device=DEV)

  def checked():
    d = env.scene["robot"].data
    qd = d.joint_vel[:, t._ids].clone()
    q = d.joint_pos[:, t._ids].clone()
    p = t.passive.unsqueeze(-1)
    kp_e = torch.where(p, torch.zeros_like(kp), kp)
    kd_e = torch.where(p, torch.ones_like(kd), kd)
    pd = kp_e * (t.q_star - q) - kd_e * qd
    ref = torch.clamp(pd, -TAU_CAP, TAU_CAP)
    step()
    f = env.sim.data.actuator_force[:, t._ctrl]
    stats["err"] = max(stats["err"], float((f - ref).abs().max()))
    stats["absmax"] = max(stats["absmax"], float(f.abs().max()))
    stats["qdmax"] = max(stats["qdmax"], float(qd.abs().max()))
    stats["sat"] += int((pd.abs() > TAU_CAP).sum())

  env.sim.step = checked
  g = torch.Generator(device="cpu").manual_seed(1)
  for _ in range(steps):
    env.step((torch.randn(n, 12, generator=g) * 3.0).to(DEV))
  env.sim.step = step
  env.close()
  return stats


def t10_torque_cap(_):
  out = []
  ok = True
  for task in (WALK, REC_STATELESS):
    s = _substep_check(task)
    ok &= s["absmax"] <= TAU_CAP + 1e-4 and s["err"] < 1e-3 and s["sat"] > 0
    out.append(
      f"{task.split('-')[-1]}: |tau|max {s['absmax']:.2f} <= 16, "
      f"|force - clip(kp(q*-q) - kd qd, +-16)| {s['err']:.1e} "
      f"({s['sat']} saturated samples, passive kp 0 / kd 1 included)"
    )
  return ok, "; ".join(out)


def t11_qd_penalty(_):
  """qd_soft_envelope = sum(relu(|qd| - 6.28)^2), weight -0.01 in every stage."""
  from minipi_getup.ftsr_ref.config.rewards import REWARD_TABLE
  from minipi_getup.ftsr_ref.mdp import rewards as R

  (w, params) = REWARD_TABLE["qd_soft_envelope"]
  qd = torch.tensor(
    [
      [6.28] + [0.0] * 11,
      [-6.28] + [0.0] * 11,
      [6.0, -5.0] + [0.0] * 10,
      [7.28] + [0.0] * 11,
      [-7.28, 8.28] + [0.0] * 10,
    ]
  )
  orig = R._qd
  R._qd = lambda env: qd
  try:
    got = R.qd_soft_envelope.compute(None, None, **params)
  finally:
    R._qd = orig
  ref = torch.tensor([0.0, 0.0, 0.0, 1.0, 5.0])
  ok = (
    params == {"limit": 6.28}
    and tuple(w) == (-0.01, -0.01, -0.01)
    and torch.allclose(got.float(), ref, atol=1e-5)
  )
  return ok, (
    f"limit {params['limit']}, weights {w}; penalty at |qd| = 6.28 / 6.0 / 7.28 / "
    f"(7.28, 8.28): {got.tolist()} (expected {ref.tolist()})"
  )


def t12_actor_layout(_):
  env = make_env(WALK, 4, play=True)
  for _ in range(5):
    env.step(torch.randn(4, 12, device=DEV) * 0.5)
  obs = env.get_observations()["actor"]
  t = term(env)
  d = env.scene["robot"].data
  cmd = env.command_manager.get_command("twist")
  ref = torch.cat(
    (
      torch.zeros(4, 3, device=DEV),
      env.scene["robot/imu_ang_vel"].data * 0.25,
      d.projected_gravity_b,
      cmd * torch.tensor([2.0, 2.0, 0.25], device=DEV),
      d.joint_pos[:, t._ids],
      d.joint_vel[:, t._ids] * 0.05,
      t.raw_action,
    ),
    -1,
  )
  gyro_vs_root = float(
    (env.scene["robot/imu_ang_vel"].data - d.root_link_ang_vel_b).abs().max()
  )
  err = float((obs - ref).abs().max())
  env.close()
  return obs.shape[-1] == 48 and err < 1e-5, (
    f"o_t 48 = [0 lin 3 | gyro*0.25 3 | g 3 | cmd*(2,2,.25) 3 | q 12 | qd*.05 12 | a 12], "
    f"err {err:.1e}; IMU gyro vs root ang vel {gyro_vs_root:.1e}"
  )


def t13_history_layout(_):
  env = make_env(WALK, 4, play=True)
  frames = []
  ok_zero = bool((env.get_observations()["policy"][:, :192] == 0).all())
  for _ in range(7):
    env.step(torch.randn(4, 12, device=DEV) * 0.5)
    o = env.get_observations()
    frames.append(o["actor"].clone())
  pol = o["policy"].view(4, 5, 48)
  ok_order = all(torch.allclose(pol[:, i], frames[-5 + i]) for i in range(5))
  ok_last = torch.allclose(o["policy"][:, -48:], o["actor"])
  env.reset(env_ids=torch.tensor([1], device=DEV))
  p = env.get_observations()["policy"].view(4, 5, 48)
  ok_reset = bool((p[1, :4] == 0).all()) and torch.allclose(p[0, -1], frames[-1][0])
  env.close()
  ok = ok_zero and ok_order and ok_last and ok_reset
  return ok, (
    f"zero-init {ok_zero}, frame-major oldest->newest {ok_order}, newest = o_t {ok_last}, "
    f"reset zeroes only reset env {ok_reset}"
  )


def t14_last_action(_):
  env = make_env(REC, 4, play=True)
  t = term(env)
  ok_passive = True
  for k in range(PASSIVE_STEPS):
    env.step(torch.ones(4, 12, device=DEV) * 2.0)
    o = env.get_observations()["actor"]
    ok_passive &= bool((t.raw_action == 0).all())
    if k < PASSIVE_STEPS - 1:
      ok_passive &= bool((o == 0).all())
  a = torch.randn(4, 12, device=DEV) * 2
  env.step(a)
  o = env.get_observations()["actor"]
  ok_act = torch.allclose(o[:, 36:48], a)  # raw (within raw clip)
  sc = torch.tensor(per_joint(ACTION_SCALE), device=DEV, dtype=torch.float)
  ok_target = torch.allclose(t.q_star, torch.clamp(sc * a, t.q_min, t.q_max))
  env.close()
  return ok_passive and ok_act and ok_target, (
    f"zero in passive window {ok_passive}; = raw-clipped action after it {ok_act}; "
    f"PD target = clip(scale * action) (no slew) {ok_target}"
  )


def t15_teacher_obs(_):
  env = make_env(WALK, 8, play=True)
  for _ in range(50):
    env.step(torch.zeros(8, 12, device=DEV))
  x = env.get_observations()["teacher"]
  fz = float(x[:, [2, 5]].sum(-1).mean())
  h = float(x[:, 6].mean())
  env.close()
  ok = x.shape[-1] == 20 and abs(abs(fz) - 1.0) < 0.1
  return ok, (
    f"dim {x.shape[-1]}; standing feet F_z sum {fz:.3f} m g (sensor sign convention); "
    f"height feature {h:.3f}"
  )


def t16_split(_):
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from minipi_getup.ftsr_ref.config.rl_cfg import recovery_runner_cfg
  from minipi_getup.ftsr_ref.rl.runner import FtsrRunner

  n_ref = int(round(0.75 * 4000))
  env = RslRlVecEnvWrapper(make_env(WALK, 8))
  r = FtsrRunner(env, asdict(recovery_runner_cfg()), None, DEV)
  ok = (
    n_ref == 3000
    and r.num_teacher == 6
    and r.teacher_mask.tolist() == [True] * 6 + [False] * 2
  )
  # Teacher rows act on E^t, student rows on E^s.
  obs = env.get_observations()
  with torch.no_grad():
    z_t, z_s, mean, *_ = r.act(obs)
    m_t = r.model.actor_mean(z_t, obs["actor"])
    m_s = r.model.actor_mean(z_s, obs["actor"])
  ok &= torch.allclose(mean[:6], m_t[:6]) and torch.allclose(mean[6:], m_s[6:])
  env.close()
  return ok, "4000 envs -> 3000 teacher rows [0, 3000); actor latent per group verified"


def t17_minibatch_mask(_):
  T, N, nt = 24, 40, 30
  st = RolloutStorage(
    T, N, {"actor": 3, "policy": 3, "teacher": 2, "critic": 2}, 2, 4, 2, "cpu"
  )
  mask = torch.arange(N) < nt
  # Mark each sample's z_s with its row id to check alignment after shuffling.
  st.z_s[:] = torch.arange(N).float().view(1, N, 1)
  ok = True
  for mb in st.minibatches(4, 3, mask):
    ok &= bool((mb["teacher"] == (mb["row"] < nt)).all())
    ok &= bool((mb["z_s"][:, 0].long() == mb["row"]).all())
  return ok, "after randperm every sample keeps its own teacher flag and stored latent"


def t18_stage(_):
  cfg = StageCfg(heights=STAGE_HEIGHTS)
  h1, h2, _ = STAGE_HEIGHTS
  n = 300
  cases = []
  for above1, above2, expect in (
    (199, 0, 0),
    (201, 0, 1),
    (300, 199, 1),
    (300, 201, 2),
  ):
    h = torch.full((n,), 0.1)
    h[:above1] = h1 + 0.01
    h[:above2] = h2 + 0.01
    stage, *_ = decide_stage(h, cfg)
    cases.append(stage == expect)
  env = make_env(REC, 16)
  env.step(torch.zeros(16, 12, device=DEV))
  st = env.extras["ftsr_stage"]
  lying = st.stage == 0 and abs(st.h_cmd - h1) < 1e-9
  lying &= abs(st.h_reward - STAGE_REWARD_HEIGHTS[0]) < 1e-9 and st.h_reward > h1
  env.close()
  env = make_env(WALK, 4)
  env.step(torch.zeros(4, 12, device=DEV))
  walk = env.extras["ftsr_stage"].stage == 2
  env.close()
  ok = all(cases) and lying and walk
  return ok, (
    f"2/3 rule at 199/201 of 300 (h1 {h1}, h2 {h2:.3f}): {cases}; lying population -> "
    f"r_u (h_cmd {h1}, reward target {st.h_reward:.4f}) {lying}; walk task fixed r_w {walk}"
  )


def t19_assist_direction(_):
  # Rotation vector maps body z onto world z for random orientations.
  q = torch.nn.functional.normalize(torch.randn(256, 4), dim=-1)
  rv = uprighting_rotvec(q)
  w, x, y, z = q.unbind(-1)
  bz = torch.stack(
    (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), -1
  )
  ang = rv.norm(dim=-1, keepdim=True).clamp(min=1e-9)
  k = rv / ang
  rot = (
    bz * torch.cos(ang)
    + torch.cross(k, bz, dim=-1) * torch.sin(ang)
    + k * (k * bz).sum(-1, keepdim=True) * (1 - torch.cos(ang))
  )
  ok_rot = bool(((rot - torch.tensor([0.0, 0.0, 1.0])).norm(dim=-1) < 1e-4).all())
  ok_yaw = bool((rv[:, 2].abs() < 1e-6).all())

  def run(assist_on: bool):
    def mutate(cfg):
      cfg.events["reset_pose"].params["by_env_index"] = True
      if not assist_on:
        cfg.actions["joint_pos"].assist = None

    env = make_env(REC, 16, mutate=mutate)
    for _ in range(PASSIVE_STEPS + 50):
      env.step(torch.zeros(16, 12, device=DEV))
    t = term(env)
    h = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    f = t.force[:, 2].clone()
    env.close()
    return h, f

  h_on, f_on = run(True)
  h_off, _ = run(False)
  ok = (
    ok_rot
    and ok_yaw
    and bool((f_on >= 0).all())
    and float(h_on.mean()) > float(h_off.mean()) + 0.02
  )
  return ok, (
    f"rotvec rights body z (256 random) {ok_rot}, no yaw {ok_yaw}; after 1 s holding "
    f"pose: base height {float(h_on.mean()):.3f} (assist, F mean {float(f_on.mean()):.1f} N, "
    f"F_max {ASSIST_F_MAX:.1f}) vs {float(h_off.mean()):.3f} m (none)"
  )


def t20_assist_cutoff(_):
  env = make_env(REC, 8)
  t = term(env)
  t_tag = t.cfg.assist.t_tag
  for _ in range(PASSIVE_STEPS + 2):
    env.step(torch.zeros(8, 12, device=DEV))
  env.common_step_counter = t_tag - 2
  env.step(torch.zeros(8, 12, device=DEV))
  before = (t.t_coeff, float(t.force.norm(dim=-1).max()))
  env.step(torch.zeros(8, 12, device=DEV))  # processed at counter t_tag - 1 ... >0
  env.step(torch.zeros(8, 12, device=DEV))  # processed at counter >= t_tag
  after_coeff = t.t_coeff
  f_after = float(t.force.abs().max())
  c_after = float(env.extras[COST_KEY].abs().max())
  xfrc = float(env.sim.data.xfrc_applied.abs().max())
  env.close()
  ok = (
    before[0] > 0
    and before[1] > 0
    and after_coeff == 0.0
    and f_after == 0.0
    and c_after == 0.0
    and xfrc == 0.0
  )
  ok &= (
    time_coeff(t_tag, t.cfg.assist) == 0.0 and time_coeff(t_tag - 1, t.cfg.assist) > 0
  )
  return ok, (
    f"t_tag = 3000 x 24 = {t_tag}: before coeff {before[0]:.2e}, |F| {before[1]:.3f} N; "
    f"after coeff {after_coeff}, |F| {f_after}, cost {c_after}, xfrc_applied {xfrc}"
  )


def t21_force_guidance(_):
  """Hand-computed 4-step, 2-env rollout (gamma 0.9, lambda 0.5, beta 0.01)."""
  g, lam, beta = 0.9, 0.5, 0.01
  r = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
  v = torch.tensor([[0.5, 0.2], [0.4, 0.1], [0.3, 0.6], [0.2, 0.0]])
  d = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
  last_v = torch.tensor([0.1, 0.3])
  cf = torch.tensor([[0.4, 0.0], [0.2, 0.8], [0.0, 0.1], [0.1, 0.0]])
  ct = torch.tensor([[0.0, 0.2], [0.1, 0.0], [0.3, 0.0], [0.0, 0.4]])

  # Hand calculation, env 0 (no done): deltas then backward recursion.
  d0 = [
    1.0 + g * 0.4 - 0.5,
    0.0 + g * 0.3 - 0.4,
    2.0 + g * 0.2 - 0.3,
    1.0 + g * 0.1 - 0.2,
  ]
  a0 = [0.0] * 4
  a0[3] = d0[3]
  a0[2] = d0[2] + g * lam * a0[3]
  a0[1] = d0[1] + g * lam * a0[2]
  a0[0] = d0[0] + g * lam * a0[1]
  # env 1: done after t=1 (no bootstrap across, recursion cut).
  d1 = [0.0 + g * 0.1 - 0.2, 1.0 - 0.1, 0.0 + g * 0.0 - 0.6, 1.0 + g * 0.3 - 0.0]
  a1 = [0.0] * 4
  a1[3] = d1[3]
  a1[2] = d1[2] + g * lam * a1[3]
  a1[1] = d1[1]
  a1[0] = d1[0] + g * lam * a1[1]
  adv_hand = torch.tensor([a0, a1]).T
  _, adv = gae(r, v, last_v, d, g, lam)
  ok_gae = torch.allclose(adv, adv_hand, atol=1e-6)

  # Cost GAE with zero baseline = lambda-discounted sums; J = gamma-discounted sums.
  def lam_sum(c, done_t1):
    out = [0.0] * 4
    run = 0.0
    for t in reversed(range(4)):
      if done_t1 and t == 1:
        run = 0.0
      run = c[t] + g * lam * run if not (done_t1 and t == 1) else c[t]
      out[t] = run
    return out

  def g_sum(c, done_t1):
    out = [0.0] * 4
    run = 0.0
    for t in reversed(range(4)):
      run = c[t] + (0.0 if (done_t1 and t == 1) else g * run)
      out[t] = run
    return out

  j_f = torch.tensor(
    [g_sum(cf[:, 0].tolist(), False), g_sum(cf[:, 1].tolist(), True)]
  ).T
  ok_j = torch.allclose(discounted_cost_to_go(cf, d, g), j_f, atol=1e-6)
  acf = torch.tensor(
    [lam_sum(cf[:, 0].tolist(), False), lam_sum(cf[:, 1].tolist(), True)]
  ).T
  act = torch.tensor(
    [lam_sum(ct[:, 0].tolist(), False), lam_sum(ct[:, 1].tolist(), True)]
  ).T
  j_t = torch.tensor(
    [g_sum(ct[:, 0].tolist(), False), g_sum(ct[:, 1].tolist(), True)]
  ).T

  def std(x):
    return (x - x.mean()) / (x.std() + 1e-8)

  hand = (
    std(adv_hand)
    - beta * (j_f + std(acf) / (1 - g))
    - beta * (j_t + std(act) / (1 - g))
  )
  costs = torch.stack((cf, ct), -1)
  got, _ = mixed_advantage(adv, costs, d, g, lam, (beta, beta))
  ok_mix = torch.allclose(got, hand, atol=1e-5)

  # Ambiguity variants (same rollout).
  hand_imm = (
    std(adv_hand) - beta * (cf + std(acf) / (1 - g)) - beta * (ct + std(act) / (1 - g))
  )
  got_imm, _ = mixed_advantage(adv, costs, d, g, lam, (beta, beta), j_mode="immediate")
  hand_raw = std(adv_hand) - beta * (j_f + acf / (1 - g)) - beta * (j_t + act / (1 - g))
  got_raw, _ = mixed_advantage(
    adv, costs, d, g, lam, (beta, beta), standardize_cost_adv=False
  )
  ok_var = torch.allclose(got_imm, hand_imm, atol=1e-5) and torch.allclose(
    got_raw, hand_raw, atol=1e-5
  )
  # After t_tag (zero costs): A_bar == standardized A exactly.
  got0, _ = mixed_advantage(adv, torch.zeros_like(costs), d, g, lam, (beta, beta))
  ok_zero = torch.allclose(got0, std(adv_hand), atol=1e-6)

  # Ambiguity experiment: penalty size vs |A| (= 1 after standardization) for the
  # paper's gamma = 0.99, beta = 0.001 / 0.02 and a cost at its maximum 1.
  g99 = 0.99
  rows = []
  for b in (0.001, 0.02):
    j_max = 1.0 / (1 - g99)
    rows.append(
      f"beta {b}: J term <= {b * j_max:.2f}, std(A_C) term {b / (1 - g99):.2f}/unit"
    )
  ok = ok_gae and ok_j and ok_mix and ok_var and ok_zero
  return ok, (
    f"GAE {ok_gae}, J cost-to-go {ok_j}, Eq.8 default {ok_mix}, variants (J immediate, "
    f"unstandardized A_C) {ok_var}, zero cost -> std(A) {ok_zero}; " + "; ".join(rows)
  )


def t22_settled_reset(_):
  n = 256

  def mutate(cfg):
    cfg.events["reset_pose"].params["by_env_index"] = True

  env = make_env(REC, n, mutate=mutate)
  t = term(env)
  d = env.scene["robot"].data
  for _ in range(PASSIVE_STEPS):
    env.step(torch.zeros(n, 12, device=DEV))
  # Next process_actions is the entry step: measure the state now.
  v = d.root_link_lin_vel_w.norm(dim=-1)
  w = d.root_link_ang_vel_w.norm(dim=-1)
  qd = d.joint_vel[:, t._ids].abs().amax(-1)
  h = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  g = d.projected_gravity_b
  pose = env.extras["ftsr_reset_pose"]
  lines = []
  ok = True
  for i, name in enumerate(mdp.FALLEN_POSE_NAMES):
    m = pose == i
    # Expected gravity direction in body frame: supine body x up -> g_x = -1? (report)
    lines.append(
      f"{name}: h {float(h[m].mean()):.3f}+-{float(h[m].std()):.3f} m, g_b "
      f"({float(g[m, 0].mean()):+.2f},{float(g[m, 1].mean()):+.2f},{float(g[m, 2].mean()):+.2f}), "
      f"|v| p95 {float(v[m].quantile(0.95)):.3f} m/s, |w| p95 {float(w[m].quantile(0.95)):.2f}, "
      f"|qd| p95 {float(qd[m].quantile(0.95)):.2f}"
    )
  ok &= float(v.quantile(0.95)) < 0.05 and float(w.quantile(0.95)) < 0.5
  env.close()
  return ok, " | ".join(lines)


def t23_random_rollout(_):
  out = []
  ok = True
  for task in (WALK, REC_STATELESS):
    for mag in (2.0, 50.0):  # 50: raw-clip-sized, bang-bang targets
      env = make_env(task, 64)
      bad = 0
      for _ in range(300):
        obs, rew, *_ = env.step(torch.randn(64, 12, device=DEV) * mag)
        bad += int(sum(int((~torch.isfinite(x)).sum()) for x in obs.values()))
        bad += int((~torch.isfinite(rew)).sum())
      env.close()
      ok &= bad == 0
      out.append(f"{task.split('-')[-1]}/action std {mag}: {bad} non-finite")
  return ok, "300 steps x 64 envs, " + ", ".join(out)


def t24_resume(_):
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from minipi_getup.ftsr_ref.config.rl_cfg import recovery_runner_cfg
  from minipi_getup.ftsr_ref.rl.runner import FtsrRunner

  cfg = asdict(recovery_runner_cfg())
  cfg["eval_every"] = 0
  tmp = tempfile.mkdtemp()
  env = RslRlVecEnvWrapper(make_env(REC, 32))
  r = FtsrRunner(env, cfg, tmp, DEV)
  r.learn(2)
  path = os.path.join(tmp, "model_2.pt")
  counter = env.unwrapped.common_step_counter
  obs = env.get_observations()
  with torch.no_grad():
    ref = r.act(obs)[2]
  coeff_ref = term(env.unwrapped).t_coeff
  stage_ref = env.unwrapped.extras["ftsr_stage"].stage
  sd = {k: v.clone() for k, v in r.model.state_dict().items()}
  opt_state = r.optimizer.state_dict()
  lr = r.lr
  env.close()

  env2 = RslRlVecEnvWrapper(make_env(REC, 32))
  r2 = FtsrRunner(env2, cfg, None, DEV)
  r2.load(path)
  ok = (
    r2.current_learning_iteration == 2 and env2.unwrapped.common_step_counter == counter
  )
  ok &= all(torch.equal(sd[k], v) for k, v in r2.model.state_dict().items())
  ok &= abs(r2.lr - lr) < 1e-12
  s1 = opt_state["state"]
  s2 = r2.optimizer.state_dict()["state"]
  ok &= all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
  with torch.no_grad():
    got = r2.act(obs)[2]
  ok &= torch.allclose(ref, got)
  env2.step(torch.zeros(32, 12, device=DEV))
  coeff2 = term(env2.unwrapped).t_coeff
  stage2 = env2.unwrapped.extras["ftsr_stage"].stage
  ok &= abs(coeff2 - time_coeff(counter, term(env2.unwrapped).cfg.assist)) < 1e-12
  r2.learn(1)
  ok &= (
    os.path.exists(os.path.join(tmp, "model_2.pt"))
    and r2.current_learning_iteration == 3
  )
  env2.close()
  return ok, (
    f"iter 2 -> 2, env counter {counter} restored, weights/Adam/lr equal, same actions on "
    f"same obs; assist coeff {coeff_ref:.5f} -> {coeff2:.5f} (one step later); stage "
    f"recomputed {stage_ref} -> {stage2}; continued to iter {r2.current_learning_iteration}"
  )


def t26_monotonic_stage(_):
  from minipi_getup.ftsr_ref.mdp.stages import next_stage

  cfg = StageCfg(heights=STAGE_HEIGHTS, monotonic=True)
  cases = {
    "A stage 0, S1 0.60 -> 0": next_stage(0, 0.60, 0.0, cfg) == 0,
    "B stage 0, S1 0.70 -> 1": next_stage(0, 0.70, 0.0, cfg) == 1,
    "C stage 1, S1 0.20 S2 0.10 -> 1": next_stage(1, 0.20, 0.10, cfg) == 1,
    "D stage 1, S2 0.70 -> 2": next_stage(1, 0.90, 0.70, cfg) == 2,
    "E stage 2, S1 0 S2 0 -> 2": next_stage(2, 0.0, 0.0, cfg) == 2,
    "one step at a time (0, S2 0.9 -> 1)": next_stage(0, 0.95, 0.90, cfg) == 1,
    "forward threshold strict (0, 2/3 -> 0)": next_stage(0, 2.0 / 3.0, 0, cfg) == 0,
  }
  # Live env: the recovery task latches; a lying population in stage 1 / 2 stays there.
  env = make_env(REC, 16)
  st = env.extras["ftsr_stage"]
  live = st.cfg.monotonic and st.cfg.heights == STAGE_HEIGHTS
  env.step(torch.zeros(16, 12, device=DEV))
  live &= st.stage == 0 and float(st.s1) < 2.0 / 3.0
  for forced in (1, 2):
    st.stage = forced
    for _ in range(3):
      env.step(torch.zeros(16, 12, device=DEV))
    live &= st.stage == forced and abs(st.h_cmd - STAGE_HEIGHTS[forced]) < 1e-9
  live &= st.transitions == []
  env.close()
  ok = all(cases.values()) and live
  return ok, f"{cases}; live lying population keeps latched stage 1 and 2: {live}"


def t27_stage_checkpoint(_):
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from minipi_getup.ftsr_ref.config.rl_cfg import recovery_runner_cfg
  from minipi_getup.ftsr_ref.rl.runner import FtsrRunner

  cfg = asdict(recovery_runner_cfg())
  cfg["eval_every"] = 0
  tmp = tempfile.mkdtemp()
  got = {}
  for saved in (1, 2):
    env = RslRlVecEnvWrapper(make_env(REC, 16))
    r = FtsrRunner(env, cfg, tmp, DEV)
    env.unwrapped.extras["ftsr_stage"].stage = saved
    path = os.path.join(tmp, f"stage_{saved}.pt")
    r.save(path)
    env.close()
    env2 = RslRlVecEnvWrapper(make_env(REC, 16))
    r2 = FtsrRunner(env2, cfg, None, DEV)
    st = env2.unwrapped.extras["ftsr_stage"]
    before = st.stage
    r2.load(path)
    after_load = st.stage
    # Lying population after resume: S1 = 0, still the saved stage.
    for _ in range(3):
      env2.step(torch.zeros(16, 12, device=DEV))
    got[saved] = (before, after_load, st.stage, float(st.s1))
    env2.close()
  # Fresh start from a walking checkpoint (weights only) begins at stage 0.
  env3 = RslRlVecEnvWrapper(make_env(REC, 16))
  r3 = FtsrRunner(env3, cfg, None, DEV)
  r3.load_weights(os.path.join(tmp, "stage_2.pt"))
  env3.step(torch.zeros(16, 12, device=DEV))
  fresh = env3.unwrapped.extras["ftsr_stage"].stage
  env3.close()
  ok = all(v[1] == k and v[2] == k for k, v in got.items()) and fresh == 0
  return ok, (
    "(fresh env stage, after load, after 3 lying steps, S1): "
    f"F saved 1 -> {got[1]}; G saved 2 -> {got[2]}; weights-only init -> stage {fresh}"
  )


def t28_stateless_v2_config(_):
  """The training task keeps the v2 method: stateless stages, thresholds, reward
  targets, Eq. 4 schedule, passive window, physics / policy timing."""
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  cfg = load_env_cfg(REC_STATELESS)
  st = cfg.events["stage_setup"].params["stage"]
  act = cfg.actions["joint_pos"]
  rl = load_rl_cfg(REC_STATELESS)
  checks = {
    "stateless": st.monotonic is False,
    "thresholds": np.allclose(st.heights, (0.19, 0.276, 0.345)),
    "reward targets": np.allclose(st.reward_heights, (0.21375, 0.276, 0.345)),
    "2/3 rule": abs(st.fraction - 2.0 / 3.0) < 1e-12,
    "assist end 3000": act.assist.end_iteration == 3000,
    "passive 2 s": act.passive_steps == 100,
    "timing 0.5 ms x 40": cfg.sim.mujoco.timestep == 0.0005 and cfg.decimation == 40,
    "no slew attr": not hasattr(act, "slew_rate"),
    "entropy 0.01 / std 1.0": rl.entropy_coef == 0.01 and rl.init_noise_std == 1.0,
    "eval task stateless": rl.eval_task == REC_STATELESS,
  }
  return all(checks.values()), str(checks)


def t29_nonfinite_guard(_):
  """A non-finite PPO minibatch is skipped (parameters unchanged and finite); three
  consecutive bad iterations stop training. Finite batches are untouched."""
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from minipi_getup.ftsr_ref.config.rl_cfg import recovery_runner_cfg
  from minipi_getup.ftsr_ref.rl.runner import FtsrRunner

  cfg = asdict(recovery_runner_cfg())
  cfg["eval_at"], cfg["eval_every"] = (), 0
  env = RslRlVecEnvWrapper(make_env(REC_STATELESS, 32))
  r = FtsrRunner(env, cfg, tempfile.mkdtemp(), DEV)
  r.learn(1)  # finite: no skips
  clean = r._consec_ppo == 0 if hasattr(r, "_consec_ppo") else True

  def _rollout(obs):
    with torch.inference_mode():
      obs = _rollout_steps(obs)
    with torch.no_grad():  # as in FtsrRunner.learn
      lv = r.model.value(r.model.teacher_encoder(obs["teacher"]), obs["critic"])
      r.storage.compute(lv, 0.99, 0.95, (0.001, 0.001), False, "cost_return", True)

  def _rollout_steps(obs):
    for _ in range(cfg["num_steps_per_env"]):
      z_t, z_s, mean, std, value = r.act(obs)
      a = mean
      step_obs = {g: obs[g] for g in ("actor", "policy", "teacher", "critic")}
      obs, rew, dones, _ = env.step(a)
      r.storage.add(
        step_obs,
        z_t=z_t,
        z_s=z_s,
        actions=a,
        log_prob=torch.distributions.Normal(mean, std).log_prob(a).sum(-1),
        mu=mean,
        sigma=std,
        rewards=rew,
        costs=torch.zeros(32, 2, device=DEV),
        dones=dones.float(),
        values=value,
      )
    return obs

  def poisoned_update():
    _rollout(env.get_observations())
    r.storage.advantages.fill_(float("nan"))
    before = {n: p.clone() for n, p in r.model.named_parameters()}
    try:
      r._update()
      raised = False
    except RuntimeError as e:
      if "persistent non-finite" not in str(e):
        raise
      raised = True
    r.storage.clear()
    same = all(torch.equal(before[n], p) for n, p in r.model.named_parameters())
    return raised, same

  raised, same = poisoned_update()
  env.close()
  ok = clean and raised and same
  return ok, (
    f"finite iteration clean {clean}; all-NaN advantages: every minibatch skipped, "
    f"parameters unchanged {same}, training stopped (RuntimeError) {raised}"
  )


PRE_FIX_COMMIT = "36c1bc6"  # last commit before the explosion fix (reference code)


def _pre_fix_module(rel: str, name: str):
  """The module ``rel`` as of PRE_FIX_COMMIT (reference for bitwise identity)."""
  import importlib.util
  import subprocess

  repo = os.path.dirname(os.path.abspath(__file__))
  src = subprocess.run(
    ["git", "-C", repo, "show", f"{PRE_FIX_COMMIT}:src/minipi_getup/ftsr_ref/{rel}"],
    capture_output=True,
    text=True,
    check=True,
  ).stdout
  path = os.path.join(tempfile.mkdtemp(), f"{name}.py")
  with open(path, "w") as f:
    f.write(src)
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def t30_explosion_source(_):
  """A non-finite physics state in one env: flagged invalid, zero wrench and cost at
  the source, nan-terminated and reset. Finite envs: the sanitation is an identity
  (``torch.where`` / ``masked_fill`` with an all-finite mask, checked bitwise on the
  clean run's tensors; bitwise PPO identity is test 32). Two separate env instances
  are not bitwise reproducible on the GPU, so the twin comparison of envs 1-7 is
  reported against the clean-vs-clean difference, not required to be 0."""
  from minipi_getup.ftsr_ref.mdp.actions import INVALID_KEY

  n = 8

  def run(poison: bool):
    torch.manual_seed(0)
    env = make_env(REC_STATELESS, n)
    term_ = env.action_manager.get_term("joint_pos")
    a = torch.zeros(n, 12, device=DEV)
    for _ in range(105):  # past the 2 s passive window: assistance active
      env.step(a)
    if poison:
      env.sim.data.qpos[0, 2] = float("nan")
    _, _, terminated, _, extras = env.step(a)
    out = {
      "cost": env.extras[COST_KEY].clone(),
      "invalid": env.extras[INVALID_KEY].clone(),
      "force": term_.force.clone(),
      "nan_term": env.termination_manager.get_term("nan").clone(),
      "terminated": terminated.clone(),
      "obs_finite": all(
        bool(torch.isfinite(v).all())
        for v in env.observation_manager.compute().values()
      ),
    }
    env.step(a)
    out["invalid_next"] = env.extras[INVALID_KEY].clone()
    out["qpos_finite_next"] = bool(torch.isfinite(env.sim.data.qpos).all())
    out["monitor"] = term_.monitor.summary(term_.joint_names)
    env.close()
    return out

  p, c, c2 = run(True), run(False), run(False)
  noise = float((c["cost"] - c2["cost"]).abs().max())
  diff = float((p["cost"][1:] - c["cost"][1:]).abs().max())
  okm = ~c["invalid"].unsqueeze(-1)
  ident = torch.equal(torch.where(okm, c["force"], 0.0), c["force"]) and torch.equal(
    c["cost"].masked_fill(c["invalid"].unsqueeze(-1), 0.0), c["cost"]
  )
  mon_ok = all(math.isfinite(v) for v in p["monitor"].values() if isinstance(v, float))
  ok = (
    bool(p["invalid"][0])
    and not bool(p["invalid"][1:].any())
    and not bool(c["invalid"].any())
    and bool(torch.isfinite(p["cost"]).all())
    and bool((p["cost"][0] == 0).all())
    and bool((c["cost"][:, 0] > 0).all())
    and bool(torch.isfinite(p["force"]).all())
    and bool(p["nan_term"][0])
    and bool(p["terminated"][0])
    and p["obs_finite"]
    and not bool(p["invalid_next"].any())
    and p["qpos_finite_next"]
    and mon_ok
    and ident
    and diff <= 10.0 * max(noise, 1e-6)
  )
  return ok, (
    f"poisoned env 0: invalid {bool(p['invalid'][0])}, cost {p['cost'][0].tolist()}, "
    f"nan-terminated {bool(p['nan_term'][0])}, obs finite {p['obs_finite']}, valid "
    f"again next step {not bool(p['invalid_next'].any())}; monitor finite {mon_ok}; "
    f"sanitation identity on finite envs {ident}; envs 1-7 vs clean twin max |diff| "
    f"{diff:.1e} (clean vs clean {noise:.1e}; clean cost min "
    f"{float(c['cost'][:, 0].min()):.3f})"
  )


def t31_invalid_storage(_):
  """Storage: all-valid rollouts give bitwise the pre-fix returns / advantages (with
  and without Eq. 8); an invalid sample with NaN reward and cost gives finite
  advantages, advantage 0 there, and results independent of its contents."""
  old = _pre_fix_module("rl/storage.py", "_ftsr_storage_prefix")
  T, N = 24, 64
  g = torch.Generator(device="cpu").manual_seed(3)

  def rnd(*shape):
    return torch.randn(*shape, generator=g).to(DEV)

  rew, val, costs = rnd(T, N), rnd(T, N), rnd(T, N, 2).abs()
  dones = (torch.rand(T, N, generator=g) < 0.05).float().to(DEV)
  last = rnd(N)
  betas = (0.001, 0.001)

  def new_compute(rew, costs, cons, valid=None):
    st = RolloutStorage(T, N, {"actor": 1}, 1, 1, 2, DEV)
    st.rewards, st.values, st.costs, st.dones = (
      rew.clone(),
      val.clone(),
      costs.clone(),
      dones.clone(),
    )
    if valid is not None:
      st.valid = valid.clone()
    info = st.compute(last, 0.99, 0.95, betas, cons, "cost_return", True)
    return st.returns, st.advantages, info

  ident = []
  for cons in (True, False):
    ret_n, adv_n, _ = new_compute(rew, costs, cons)
    ret_o, adv_r = old.gae(rew, val, last, dones, 0.99, 0.95)
    if cons:
      adv_o, _ = old.mixed_advantage(
        adv_r, costs, dones, 0.99, 0.95, betas, "cost_return", True
      )
    else:
      adv_o = old.standardize(adv_r)
    ident.append(torch.equal(ret_n, ret_o) and torch.equal(adv_n, adv_o))

  t0, k = 10, 3
  valid = torch.ones(T, N, dtype=torch.bool, device=DEV)
  valid[t0, k] = False
  r1, c1 = rew.clone(), costs.clone()
  r1[t0, k], c1[t0, k] = float("nan"), float("nan")  # explosion: flagged by guard 2
  _, adv1, info1 = new_compute(r1, c1, True)
  r2, c2 = rew.clone(), costs.clone()
  r2[t0, k], c2[t0, k] = 123.0, 45.0  # any contents, flagged by the runner
  _, adv2, info2 = new_compute(r2, c2, True, valid)
  # Other envs: unstandardized reward GAE unchanged; env k truncated at t0.
  _, a_masked = gae(rew, val, last, dones, 0.99, 0.95, valid)
  _, a_plain = gae(rew, val, last, dones, 0.99, 0.95)
  others = torch.arange(N, device=DEV) != k
  ok = (
    all(ident)
    and bool(torch.isfinite(adv1).all())
    and float(adv1[t0, k]) == 0.0
    and torch.equal(adv1, adv2)
    and info1["invalid_samples"] == 1.0
    and info1["nonfinite_cost_samples"] == 1.0
    and torch.equal(a_masked[:, others], a_plain[:, others])
    and torch.equal(a_masked[t0 + 1 :, k], a_plain[t0 + 1 :, k])
  )
  return ok, (
    f"all valid = pre-fix ({PRE_FIX_COMMIT}) bitwise: Eq. 8 {ident[0]}, plain "
    f"{ident[1]}; NaN sample -> advantages finite {bool(torch.isfinite(adv1).all())}, "
    f"A = {float(adv1[t0, k])} there, independent of its contents "
    f"{torch.equal(adv1, adv2)}, guard counts {info1['invalid_samples']:.0f} / "
    f"{info1['nonfinite_cost_samples']:.0f}"
  )


def t32_ppo_identity(_):
  """PPO: on an all-valid rollout the update equals the pre-fix ``_update`` bitwise;
  with an invalid sample the update is finite and independent of its contents."""
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from minipi_getup.ftsr_ref.config.rl_cfg import recovery_runner_cfg
  from minipi_getup.ftsr_ref.rl.runner import FtsrRunner

  old = _pre_fix_module("rl/runner.py", "_ftsr_runner_prefix")
  cfg = asdict(recovery_runner_cfg())
  cfg["eval_at"], cfg["eval_every"] = (), 0
  n = 32
  env = RslRlVecEnvWrapper(make_env(REC_STATELESS, n))
  r = FtsrRunner(env, cfg, None, DEV)
  st = r.storage
  obs = env.get_observations()
  g = torch.Generator(device="cpu").manual_seed(5)
  with torch.inference_mode():
    for _ in range(cfg["num_steps_per_env"]):
      z_t, z_s, mean, std, value = r.act(obs)
      a = mean + std * torch.randn(mean.shape, generator=g).to(DEV)
      step_obs = {k: obs[k] for k in ("actor", "policy", "teacher", "critic")}
      obs, rew, dones, _ = env.step(a)
      st.add(
        step_obs,
        z_t=z_t,
        z_s=z_s,
        actions=a,
        log_prob=torch.distributions.Normal(mean, std).log_prob(a).sum(-1),
        mu=mean,
        sigma=std,
        rewards=rew,
        costs=torch.rand(n, 2, generator=g).to(DEV),
        dones=dones.float(),
        values=value,
      )
    lv = r.model.value(r.model.teacher_encoder(obs["teacher"]), obs["critic"])
  env.close()
  raw = {k: getattr(st, k).clone() for k in ("rewards", "costs", "values")}
  model0 = copy.deepcopy(r.model.state_dict())
  opt0 = copy.deepcopy(r.optimizer.state_dict())
  lr0 = r.lr
  rng0 = (torch.get_rng_state(), torch.cuda.get_rng_state_all())

  def run(update, poison=None):
    for k_, v in raw.items():
      setattr(st, k_, v.clone())
    st.valid.fill_(True)
    if poison is not None:
      poison()
    st.compute(lv.clone(), 0.99, 0.95, (0.001, 0.001), True, "cost_return", True)
    r.model.load_state_dict(model0)
    r.optimizer.load_state_dict(copy.deepcopy(opt0))
    r.lr = lr0
    for grp in r.optimizer.param_groups:
      grp["lr"] = lr0
    torch.set_rng_state(rng0[0])
    torch.cuda.set_rng_state_all(rng0[1])
    update(r)
    return {k_: p.detach().clone() for k_, p in r.model.named_parameters()}, r.lr

  pa, lra = run(FtsrRunner._update)
  pb, lrb = run(old.FtsrRunner._update)
  same = lra == lrb and all(torch.equal(pa[k_], pb[k_]) for k_ in pa)

  def poison_nan():
    st.rewards[5, 2] = float("nan")
    st.costs[5, 2] = float("nan")

  def poison_garbage():
    st.valid[5, 2] = False
    st.rewards[5, 2] = 7.0
    st.costs[5, 2] = 3.0

  pc, _ = run(FtsrRunner._update, poison_nan)
  pd, _ = run(FtsrRunner._update, poison_garbage)
  finite = all(bool(torch.isfinite(v).all()) for v in pc.values())
  indep = all(torch.equal(pc[k_], pd[k_]) for k_ in pc)
  changed = any(not torch.equal(pc[k_], model0[k_]) for k_ in pc if k_ in model0)
  ok = same and finite and indep and changed
  return ok, (
    f"all-valid update = pre-fix ({PRE_FIX_COMMIT}) _update bitwise {same} (lr "
    f"{lra:.2e} / {lrb:.2e}); invalid sample: parameters finite {finite}, updated "
    f"{changed}, independent of its contents {indep}"
  )


def t25_onnx(_):
  from minipi_getup.ftsr_ref.export import export_onnx, onnx_parity
  from minipi_getup.ftsr_ref.rl.modules import FtsrModel

  torch.manual_seed(0)
  model = FtsrModel(
    {"actor": 48, "policy": 240, "teacher": 20, "critic": 50}, 12
  ).eval()
  path = os.path.join(tempfile.mkdtemp(), "policy.onnx")
  export_onnx(model, path)
  err = onnx_parity(model, path, n=32)
  import onnx

  m = onnx.load(path)
  ins = [
    (i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input
  ]
  outs = [
    (o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
    for o in m.graph.output
  ]
  ok = err < 1e-5 and ins == [("obs", [1, 240])] and outs == [("actions", [1, 12])]
  return ok, f"inputs {ins}, outputs {outs}, max |onnx - torch| {err:.2e}"


def _rate_limiter_case(task, wrap: bool):
  """Drive 8 envs with large random actions (forced reset of envs 0-1 mid-run) and
  compare q* with the reference limiter formula at every policy step."""
  from minipi_getup.ftsr_ref.limiter_eval import _install_limiter

  lim = 0.3

  def mut(cfg):
    cfg.actions["joint_pos"].passive_steps = 3

  env = make_env(task, 8, play=True, mutate=mut)
  t = term(env)
  robot = env.scene["robot"]
  if wrap:
    _install_limiter(t, robot, lim)
  gen = torch.Generator(device=DEV).manual_seed(0)
  prev = None
  out = {"mismatch": 0.0, "max_step": 0.0, "entries": 0, "raw": 0.0, "obs": 0.0}
  for k in range(40):
    if k == 20:
      env.reset(env_ids=torch.tensor([0, 1], device=DEV))
    a = 3.0 * torch.randn(8, 12, generator=gen, device=DEV)
    was = t.entered.clone()
    qm = torch.clamp(robot.data.joint_pos[:, t._ids], t.q_min, t.q_max)
    obs, *_ = env.step(a)
    ac = torch.clamp(a, -RAW_CLIP, RAW_CLIP)
    q_clip = torch.clamp(t.default + t.scale * ac, t.q_min, t.q_max)
    pas = t.passive.unsqueeze(-1)
    entry = t.entered & ~was
    base = qm if prev is None else torch.where(pas | entry.unsqueeze(-1), qm, prev)
    exp = base + torch.clamp(torch.where(pas, qm, q_clip) - base, -lim, lim)
    out["mismatch"] = max(out["mismatch"], float((t.q_star - exp).abs().max()))
    act = ~t.passive
    if act.any():
      d = (t.q_star - base).abs()[act]
      out["max_step"] = max(out["max_step"], float(d.max()))
      out["raw"] = max(out["raw"], float((t.raw_action - ac).abs()[act].max()))
      o = obs["actor"][:, 36:48]
      out["obs"] = max(out["obs"], float((o - t.raw_action).abs()[act].max()))
    out["entries"] += int(entry.sum())
    prev = t.q_star.clone()
  env.close()
  return out


def t33_rate_limiter(_):
  """Training limiter (task cfg) and evaluation wrapper both equal the reference
  q*_t = q*_prev + clip(clip(q_cmd) - q*_prev, +-0.3) at every step, initialized from
  the clipped measured pose in the passive window / first actuated step, per env
  across a forced reset; raw action and last_action unchanged."""
  from minipi_getup.ftsr_ref.config import LIMIT_TASK

  res = {
    "cfg": _rate_limiter_case(LIMIT_TASK, wrap=False),
    "wrapper": _rate_limiter_case(REC_STATELESS, wrap=True),
  }
  ok = all(
    r["mismatch"] == 0.0
    and r["max_step"] <= 0.3 + 1e-6
    and r["entries"] == 10  # 8 envs + 2 re-entries after the forced reset
    and r["raw"] == 0.0
    and r["obs"] == 0.0
    for r in res.values()
  )
  return ok, str(res)


def t34_limiter_config(_):
  """Limiter disabled by default; the limiter task differs from the stateless task
  only in the limiter (train and play) and the evaluation schedule; evaluating it
  with the wrapper as well is refused (no double application)."""
  import argparse as ap_

  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from minipi_getup.ftsr_ref import limiter_eval
  from minipi_getup.ftsr_ref.config import LIMIT_TASK

  checks = {}
  for task in (WALK, REC, REC_STATELESS):
    for play in (False, True):
      lim = load_env_cfg(task, play=play).actions["joint_pos"].target_rate_limit
      checks[f"{task[15:]} play={play} disabled"] = lim == 0.0
  for play in (False, True):
    a = copy.deepcopy(load_env_cfg(LIMIT_TASK, play=play))
    b = load_env_cfg(REC_STATELESS, play=play)
    checks[f"limit task play={play} = 0.3"] = (
      a.actions["joint_pos"].target_rate_limit == 0.3
    )
    a.actions["joint_pos"].target_rate_limit = 0.0
    # repr up to object addresses (each cfg build creates new robot-spec lambdas)
    same = re.sub(r" at 0x[0-9a-f]+", "", repr(a)) == re.sub(
      r" at 0x[0-9a-f]+", "", repr(b)
    )
    checks[f"limit task play={play} otherwise identical"] = same
  ra, rb = load_rl_cfg(LIMIT_TASK), copy.deepcopy(load_rl_cfg(REC_STATELESS))
  checks["eval schedule"] = ra.eval_every == 250 and ra.eval_task == LIMIT_TASK
  rb.eval_every, rb.eval_at, rb.eval_task = ra.eval_every, ra.eval_at, ra.eval_task
  checks["rl cfg otherwise identical"] = repr(ra) == repr(rb)
  args = ap_.Namespace(
    task=LIMIT_TASK, limit=0.3, checkpoint="", seed=2150, per_pose=1, steps=1
  )
  try:
    limiter_eval.rollout(args)
    checks["double application refused"] = False
  except SystemExit:
    checks["double application refused"] = True
  except Exception:
    checks["double application refused"] = False
  return all(checks.values()), str(checks)


TESTS = [
  (1, "joint order = deploy order", t01_joint_order),
  (2, "joint signs / default pose", t02_signs_default),
  (3, "joint limits + target clip", t03_joint_limits),
  (4, "hardware PD gains, no armature", t04_pd_gains),
  (5, "sim policy timing", t05_timing),
  (6, "target hold (ZOH) within a policy step", t06_target_hold),
  (7, "action -> q_target mapping (no slew), targets in ranges", t07_action_mapping),
  (8, "action clipping", t08_action_clip),
  (9, "no torque-speed derating", t09_no_derating),
  (10, "PD torque = clip(PD, +-16) <= cap (substeps)", t10_torque_cap),
  (11, "qd soft penalty 6.28", t11_qd_penalty),
  (12, "actor observation layout", t12_actor_layout),
  (13, "student history layout", t13_history_layout),
  (14, "last_action semantics", t14_last_action),
  (15, "teacher observation", t15_teacher_obs),
  (16, "teacher / student split", t16_split),
  (17, "minibatch teacher mask after shuffle", t17_minibatch_mask),
  (18, "stage transition", t18_stage),
  (19, "assist direction", t19_assist_direction),
  (20, "assist cutoff", t20_assist_cutoff),
  (21, "force-guidance numerical GAE (Eq. 5-8)", t21_force_guidance),
  (22, "settled reset", t22_settled_reset),
  (23, "random rollout without NaN", t23_random_rollout),
  (24, "checkpoint resume", t24_resume),
  (25, "PyTorch -> ONNX equivalence", t25_onnx),
  (26, "monotonic stage latch", t26_monotonic_stage),
  (27, "stage persists in checkpoints", t27_stage_checkpoint),
  (28, "training task keeps the v2 method", t28_stateless_v2_config),
  (29, "non-finite PPO guard", t29_nonfinite_guard),
  (30, "physics explosion: source sanitation and reset", t30_explosion_source),
  (31, "invalid samples in storage (second guard)", t31_invalid_storage),
  (32, "PPO identical on finite rollouts, invalid excluded", t32_ppo_identity),
  (33, "target-rate limiter: training = evaluation = reference", t33_rate_limiter),
  (34, "limiter config: default off, single application", t34_limiter_config),
]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--only", default="")
  args = ap.parse_args()
  only = {int(x) for x in args.only.split(",") if x}
  t0 = time.time()
  for num, name, fn in TESTS:
    if only and num not in only:
      continue
    try:
      ok, info = fn(None)
    except Exception:
      ok, info = False, traceback.format_exc()
    record(num, name, bool(ok), info)
  n_ok = sum(r[2] for r in RESULTS)
  print(f"\n{n_ok}/{len(RESULTS)} passed in {time.time() - t0:.0f} s")
  assert math.isfinite(MINIPI_MASS) and MINIPI_WEIGHT > 0


if __name__ == "__main__":
  main()
