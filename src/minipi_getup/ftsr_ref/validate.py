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
  H_CONSERVATIVE,
  H_LOOSE,
  JOINT_NAMES,
  JOINT_RANGES,
  KD,
  KP,
  MINIPI_MASS,
  MINIPI_WEIGHT,
  MotorEnvelopeCfg,
)
from minipi_getup.ftsr_ref.mdp.actions import COST_KEY, motor_envelope
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


def _reference_pipeline(a, q_prev, default, scale, lo, hi, slew):
  a_c = np.clip(a, -RAW_CLIP, RAW_CLIP)
  q_cmd = np.clip(default + scale * a_c, lo, hi)
  return np.clip(q_cmd, q_prev - slew, q_prev + slew), a_c


def t07_action_mapping(_):
  env = make_env(WALK, 8)
  t = term(env)
  scale = per_joint(ACTION_SCALE)
  ok_scale = np.allclose(t.scale.cpu().numpy(), scale)
  lo = np.array([JOINT_RANGES[n][0] for n in JOINT_NAMES])
  hi = np.array([JOINT_RANGES[n][1] for n in JOINT_NAMES])
  slew = 3.0 * 0.02
  gen = torch.Generator(device="cpu").manual_seed(0)
  q_prev = None
  worst = 0.0
  max_step = 0.0
  entry_err = 0.0
  entry = np.ones(8, dtype=bool)
  n_entries = 0
  for k in range(40):
    q = env.scene["robot"].data.joint_pos[:, t._ids].cpu().numpy()
    mag = [0.05, 1.0, 5.0, 80.0][k % 4]
    a = (torch.randn(8, 12, generator=gen) * mag).numpy()
    if q_prev is None:
      q_prev = np.zeros_like(q)
    # Entry (first step, or first step after a reset): measured pose, clipped.
    q_prev[entry] = np.clip(q[entry], lo, hi)
    ref, _ = _reference_pipeline(a, q_prev, 0.0, scale, lo, hi, slew)
    _, _, term_buf, to_buf, _ = env.step(torch.tensor(a, device=DEV, dtype=torch.float))
    got = t.q_star.cpu().numpy()
    if k == 0:
      entry_err = float(np.abs(got - ref).max())
    worst = max(worst, float(np.abs(got - ref).max()))
    max_step = max(max_step, float(np.abs(got - q_prev).max()))
    n_entries += int(entry.sum())
    q_prev = got
    entry = (term_buf | to_buf).cpu().numpy()
  # Small actions within the slew reach q_default + scale * a exactly.
  env.reset()
  for _ in range(60):
    env.step(torch.zeros(8, 12, device=DEV))
  a = torch.full((8, 12), 0.05, device=DEV)
  env.step(a)
  exact = float((t.q_star - torch.tensor(scale, device=DEV) * 0.05).abs().max())
  env.close()
  ok = ok_scale and worst < 1e-5 and max_step <= slew + 1e-6 and exact < 1e-5
  return ok, (
    f"per-joint scale {scale[:6].tolist()}; |sim - reference pipeline| max {worst:.1e} "
    f"(entry step {entry_err:.1e}, {n_entries} entries incl. resets); max |q*_t - q*_t-1| "
    f"{max_step:.4f} <= {slew}; "
    f"small action exact {exact:.1e}"
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


def t09_torque_speed_curve(_):
  m = H_CONSERVATIVE
  w0 = m.omega0
  qd = torch.tensor([0.0, 1.0, 1.869, 3.0, 5.24, w0, 9.0, -3.0, -w0, -20.0])
  lo, hi = motor_envelope(qd, m)
  hand = {  # index: hand-computed (tau_min, tau_max)
    0: (-16.0, 16.0),
    3: (-16.0, 21 * (1 - 3 / w0)),  # 12.98 Nm at the 3 rad/s envelope
    4: (-16.0, 21 * (1 - 5.24 / w0)),  # 6.98 Nm at the rated speed
    5: (-16.0, 0.0),  # no-load speed
    6: (-16.0, 21 * (1 - 9.0 / w0)),  # beyond no-load: braking only
    7: (-(21 * (1 - 3 / w0)), 16.0),
    # -20 rad/s, beyond omega0 (1 + cap / stall) = 13.8 rad/s: the linear motor can
    # only brake (generator regime), at the cap: tau_min = tau_max = +16.
    9: (16.0, 16.0),
  }
  ok = True
  for i, (l_ref, h_ref) in hand.items():
    ok &= abs(float(lo[i]) - l_ref) < 1e-4 and abs(float(hi[i]) - h_ref) < 1e-4
  return ok, (
    f"H-conservative: tau_max(0)=16, (3 rad/s)={float(hi[3]):.2f}, (5.24)={float(hi[4]):.2f}, "
    f"(7.85)={float(hi[5]):.2f}, (9.0)={float(hi[6]):.2f}; braking capped at -16"
  )


def _substep_check(task, motor: MotorEnvelopeCfg, steps=150, n=32, slew=None):
  def mutate(cfg):
    cfg.actions["joint_pos"].motor = copy.deepcopy(motor)
    if slew is not None:
      cfg.actions["joint_pos"].slew_rate = slew

  env = make_env(task, n, mutate=mutate)
  t = term(env)
  step = env.sim.step
  stats = {"viol": 0.0, "absmax": 0.0, "env_err": 0.0, "qdmax": 0.0}
  kp = torch.tensor([KP[n[2:].replace("_joint", "")] for n in JOINT_NAMES], device=DEV)
  kd = torch.tensor([KD[n[2:].replace("_joint", "")] for n in JOINT_NAMES], device=DEV)

  def checked():
    d = env.scene["robot"].data
    qd = d.joint_vel[:, t._ids].clone()
    q = d.joint_pos[:, t._ids].clone()
    lo, hi = motor_envelope(qd, motor)
    p = t.passive.unsqueeze(-1)
    kp_e = torch.where(p, torch.zeros_like(kp), kp)
    kd_e = torch.where(p, torch.ones_like(kd), kd)
    ref = torch.clamp(kp_e * (t.q_star - q) - kd_e * qd, lo, hi)
    step()
    f = env.sim.data.actuator_force[:, t._ctrl]
    # Force exactly = clip(PD, envelope(qd at step start)) (explicit force part).
    stats["env_err"] = max(stats["env_err"], float((f - ref).abs().max()))
    stats["viol"] = max(stats["viol"], float((lo - f).max()), float((f - hi).max()))
    stats["absmax"] = max(stats["absmax"], float(f.abs().max()))
    stats["qdmax"] = max(stats["qdmax"], float(qd.abs().max()))

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
  for task in (WALK, REC):
    for motor in (H_CONSERVATIVE, H_LOOSE):
      s = _substep_check(task, motor)
      ok &= (
        s["absmax"] <= motor.tau_cap + 1e-4
        and s["viol"] <= 1e-4
        and s["env_err"] < 1e-3
      )
      out.append(
        f"{task.split('-')[-1]}/{motor.name}: |tau|max {s['absmax']:.2f}, "
        f"envelope violation {s['viol']:.1e}, |force - clip(PD, env)| {s['env_err']:.1e}"
      )
  return ok, "; ".join(out)


def t11_no_load_speed(_):
  """Robot floating without gravity or contacts; square-wave targets on single joints:
  the actuator alone drives them, so joint speed must saturate at omega0 under
  H-conservative and go beyond it under H-loose."""
  joints = (
    "r_calf_joint",
    "r_thigh_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
  )
  peaks: dict[tuple[str, str, bool], float] = {}
  for motor in (H_CONSERVATIVE, H_LOOSE):
    for slew in (False, True):
      for jn in joints:

        def mutate(cfg, motor=motor, slew=slew):
          cfg.actions["joint_pos"].motor = copy.deepcopy(motor)
          if not slew:
            cfg.actions["joint_pos"].slew_rate = 0.0
          cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
          cfg.sim.mujoco.disableflags = ("contact",)
          cfg.terminations.pop("fell")

        env = make_env(WALK, 1, play=True, mutate=mutate)
        t = term(env)
        j = JOINT_NAMES.index(jn)
        lo, hi = JOINT_RANGES[jn]
        box = {"peak": 0.0}

        def logged(env=env, step=env.sim.step, col=int(t._ids[j]), box=box):
          step()
          qd = float(env.scene["robot"].data.joint_vel[:, col].abs().max())
          box["peak"] = max(box["peak"], qd)

        step = env.sim.step
        env.sim.step = logged
        for k in range(60):
          a = torch.zeros(1, 12, device=DEV)
          a[:, j] = (0.9 * hi if (k // 15) % 2 == 0 else 0.9 * lo) / float(t.scale[j])
          env.step(a)
        env.sim.step = step
        env.close()
        peaks[(motor.name, jn, slew)] = box["peak"]
  w0 = H_CONSERVATIVE.omega0
  cons_raw = [peaks[("H-conservative", j, False)] for j in joints]
  loose_raw = [peaks[("H-loose", j, False)] for j in joints]
  # Calf / hip yaw / ankle pitch saturate at omega0; the ankle roll (4.4e-4 kg m^2)
  # overshoots on raw target jumps by the first physics step's PD acceleration.
  ok = all(p <= 1.03 * w0 for p in cons_raw[:3]) and cons_raw[3] <= 1.3 * w0
  ok &= all(lp > cp + 0.5 for lp, cp in zip(loose_raw[:3], cons_raw[:3], strict=True))
  rows = [
    f"{j.split('_', 1)[1].replace('_joint', '')}: cons {peaks[('H-conservative', j, False)]:.2f}"
    f"/slew {peaks[('H-conservative', j, True)]:.2f}, loose {peaks[('H-loose', j, False)]:.2f}"
    for j in joints
  ]
  return ok, f"peak |qd| rad/s (omega0 {w0:.2f}, raw jumps / with slew): " + "; ".join(
    rows
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
  ok_act = torch.allclose(o[:, 36:48], a)  # raw (within raw clip), before slew
  ok_differs = not torch.allclose(
    t.q_star, torch.tensor(per_joint(ACTION_SCALE), device=DEV, dtype=torch.float) * a
  )
  env.close()
  return ok_passive and ok_act and ok_differs, (
    f"zero in passive window {ok_passive}; = raw-clipped action after it {ok_act}; "
    f"not the slewed target {ok_differs}"
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
  for task in (WALK, REC):
    for motor in (H_CONSERVATIVE, H_LOOSE):

      def mutate(cfg, motor=motor):
        cfg.actions["joint_pos"].motor = copy.deepcopy(motor)

      env = make_env(task, 64, mutate=mutate)
      bad = 0
      for _ in range(300):
        obs, rew, *_ = env.step(torch.randn(64, 12, device=DEV) * 2)
        bad += int(sum(int((~torch.isfinite(x)).sum()) for x in obs.values()))
        bad += int((~torch.isfinite(rew)).sum())
      env.close()
      ok &= bad == 0
      out.append(f"{task.split('-')[-1]}/{motor.name}: {bad} non-finite")
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


TESTS = [
  (1, "joint order = deploy order", t01_joint_order),
  (2, "joint signs / default pose", t02_signs_default),
  (3, "joint limits + target clip", t03_joint_limits),
  (4, "hardware PD gains, no armature", t04_pd_gains),
  (5, "sim policy timing", t05_timing),
  (6, "target hold (ZOH) within a policy step", t06_target_hold),
  (7, "action -> q_target mapping + slew + entry init", t07_action_mapping),
  (8, "action clipping", t08_action_clip),
  (9, "torque-speed curve", t09_torque_speed_curve),
  (10, "torque <= cap and inside envelope (substeps)", t10_torque_cap),
  (11, "no-load-speed behavior", t11_no_load_speed),
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
