"""Deterministic evaluation of an FTSR reference checkpoint (simulation only).

Recovery (``--task ...-Recovery``): play cfg (no noise, **no assistance**), the four
fallen poses assigned by env index, one 20 s episode per env (2 s passive + 18 s
actuated). Reported per pose, never only in aggregate:

- success: base above 0.9 x stance (0.31 m) and within 18 deg of upright, held over
  the last 3 s of the episode; time to stand (first 1 s continuous standing);
  falls after first standing.
- physical metrics at physics-step resolution (0.5 ms), actuated time only: torque
  mean / p95 / p99 / max per joint, fraction near the 16 Nm cap, fraction and
  longest run above the candidate rated torque (6 Nm), joint speed mean / p95 /
  p99 / max per joint, fractions above 3 and 4 rad/s, episodes containing > 4 rad/s,
  target tracking error, fraction above the 6.28 rad/s soft limit, mechanical power, base vertical speed, base
  roll/pitch rate, joint-limit margin, peak foot contact force.

Walk (``--task ...-Walk``): fixed command grid (vx, wz) incl. zero, 10 s episodes;
reports tracking (mean measured vs commanded), correlation, fall rate, foot
touchdowns per second and air time (stepping), zero-command drift.

Usage:
  python -m minipi_getup.ftsr_ref.evaluate --checkpoint RUN/model_N.pt \
      --task Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless [--out eval.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os

import torch

import minipi_getup  # noqa: F401
from minipi_getup.ftsr_ref.config.env_cfg import COMMAND_RANGES, STAGE_HEIGHTS
from minipi_getup.ftsr_ref.config.robot import JOINT_NAMES, TAU_CAP, TAU_RATED_REPORT
from minipi_getup.ftsr_ref.export import load_model
from minipi_getup.ftsr_ref.mdp.resets import FALLEN_POSE_NAMES, POSE_KEY
from minipi_getup.ftsr_ref.rl.modules import StudentPolicy

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
STAND_H = 0.9 * STAGE_HEIGHTS[2]
STAND_UP = math.cos(math.radians(18.0))


def build_env(task: str, n: int, mutate=None, assist_tc=None):
  """Play cfg (no noise, no assistance). ``assist_tc``: diagnostic only, re-enables
  the training Eq. 4 assistance frozen near that time coefficient (the env step
  counter is set to (1 - tc) t_tag; it drifts by < 0.02 over one episode)."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = copy.deepcopy(load_env_cfg(task, play=True))
  cfg.scene.num_envs = n
  assert cfg.actions["joint_pos"].assist is None
  if assist_tc is not None:
    train_assist = load_env_cfg(task, play=False).actions["joint_pos"].assist
    cfg.actions["joint_pos"].assist = copy.deepcopy(train_assist)
  if mutate is not None:
    mutate(cfg)
  env = ManagerBasedRlEnv(cfg, device=DEV)
  env.reset()
  if assist_tc is not None:
    t_tag = cfg.actions["joint_pos"].assist.t_tag
    env.common_step_counter = int(round((1.0 - assist_tc) * t_tag))
  return env


# Ground-contact groups for the support classification (collision geoms only).
SUPPORT_GROUPS = ("foot", "calf", "thigh", "hip", "base")


class SupportTracker:
  """Which body parts touch the ground (lowest geom point < 1 cm), per env step."""

  def __init__(self, env):
    import mujoco

    mm = env.sim.mj_model
    ids, grp = [], []
    for g in range(mm.ngeom):
      if mm.geom_contype[g] == 0 and mm.geom_conaffinity[g] == 0:
        continue
      if mm.geom_type[g] not in (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_SPHERE,
      ):
        continue
      name = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      for i, key in enumerate(SUPPORT_GROUPS):
        if key in name:
          ids.append(g)
          grp.append(i)
          break
    self.env = env
    self.ids = torch.tensor(ids, device=DEV)
    self.grp = torch.nn.functional.one_hot(
      torch.tensor(grp, device=DEV), len(SUPPORT_GROUPS)
    ).float()
    size = torch.tensor(mm.geom_size[ids], device=DEV, dtype=torch.float32)
    caps = torch.tensor(mm.geom_type[ids] == mujoco.mjtGeom.mjGEOM_CAPSULE, device=DEV)
    self.radius = size[:, 0]
    self.half = torch.where(caps, size[:, 1], torch.zeros_like(size[:, 1]))

  def touching(self) -> torch.Tensor:
    """(N, groups) bool."""
    d = self.env.sim.data
    pos = d.geom_xpos[:, self.ids]
    axz = d.geom_xmat[:, self.ids].reshape(pos.shape[0], -1, 9)[..., 8]
    low = pos[..., 2] - self.half * axz.abs() - self.radius
    low = low - self.env.scene.env_origins[:, 2:3]
    return ((low < 0.01).float() @ self.grp) > 0


class Recorder:
  """Physics-step statistics per group of envs (e.g. per pose)."""

  def __init__(
    self, env, groups: torch.Tensor, n_groups: int, tau_rated: float, tau_cap: float
  ):
    self.env, self.groups, self.ng = env, groups, n_groups
    self.t = env.action_manager.get_term("joint_pos")
    self.rated, self.cap = tau_rated, tau_cap
    n, nj = env.num_envs, len(JOINT_NAMES)
    dev = env.device
    self.tau_bins = torch.zeros(n_groups, nj, 2000, device=dev)  # 0.01 Nm bins
    self.qd_bins = torch.zeros(n_groups, nj, 3000, device=dev)  # 0.01 rad/s bins
    self.n = torch.zeros(n_groups, device=dev)
    self.tau_sum = torch.zeros(n_groups, nj, device=dev)
    self.qd_sum = torch.zeros(n_groups, nj, device=dev)
    self.tau_max = torch.zeros(n_groups, nj, device=dev)
    self.qd_max = torch.zeros(n_groups, nj, device=dev)
    self.near_cap = torch.zeros(n_groups, device=dev)
    self.above_rated = torch.zeros(n_groups, device=dev)
    self.run_rated = torch.zeros(n, nj, device=dev)
    self.longest_rated = torch.zeros(n, nj, device=dev)
    self.ep_qd_max = torch.zeros(n, device=dev)
    self.err_max = torch.zeros(n, device=dev)
    self.err_sum = torch.zeros(n, device=dev)
    self.power_sum = torch.zeros(n, device=dev)
    self.vz_max = torch.zeros(n, device=dev)
    self.wxy_max = torch.zeros(n, device=dev)
    self.margin_min = torch.full((n,), 1e9, device=dev)
    self.contact_max = torch.zeros(n, device=dev)
    self.samples = torch.zeros(n, device=dev)
    self.dt = env.physics_dt

  def __call__(self) -> None:
    env, t = self.env, self.t
    act = ~t.passive
    if not bool(act.any()):
      return
    d = env.scene["robot"].data
    tau = env.sim.data.actuator_force[:, t._ctrl]
    qd = d.joint_vel[:, t._ids]
    q = d.joint_pos[:, t._ids]
    a = act.float()
    at = tau.abs()
    aq = qd.abs()
    nj = at.shape[1]
    g = self.groups[act]
    jj = torch.arange(nj, device=at.device)
    base = g.unsqueeze(-1) * nj + jj
    ti = (at[act] / 0.01).long().clamp(max=1999) + base * 2000
    qi = (aq[act] / 0.01).long().clamp(max=2999) + base * 3000
    self.tau_bins += torch.bincount(ti.flatten(), minlength=self.ng * nj * 2000).view(
      self.ng, nj, 2000
    )
    self.qd_bins += torch.bincount(qi.flatten(), minlength=self.ng * nj * 3000).view(
      self.ng, nj, 3000
    )
    onehot = torch.nn.functional.one_hot(g, self.ng).float()  # (B, G)
    self.n += onehot.sum(0)
    self.tau_sum += onehot.T @ at[act]
    self.qd_sum += onehot.T @ aq[act]
    big = torch.full((self.ng, at.shape[0], nj), 0.0, device=at.device)
    gi = self.groups.unsqueeze(0) == torch.arange(self.ng, device=at.device).unsqueeze(
      1
    )
    gi = gi & act.unsqueeze(0)
    self.tau_max = torch.maximum(
      self.tau_max, torch.where(gi.unsqueeze(-1), at, big).amax(1)
    )
    self.qd_max = torch.maximum(
      self.qd_max, torch.where(gi.unsqueeze(-1), aq, big).amax(1)
    )
    self.near_cap += onehot.T @ (at[act] >= 0.95 * self.cap).float().sum(-1)
    self.above_rated += onehot.T @ (at[act] > self.rated).float().sum(-1)
    over = (at > self.rated) & act.unsqueeze(-1)
    self.run_rated = torch.where(
      over, self.run_rated + self.dt, torch.zeros_like(self.run_rated)
    )
    self.longest_rated = torch.maximum(self.longest_rated, self.run_rated)
    self.ep_qd_max = torch.maximum(self.ep_qd_max, aq.amax(-1) * a)
    err = (t.q_star - q).abs().amax(-1)
    self.err_max = torch.maximum(self.err_max, err * a)
    self.err_sum += err * a
    self.power_sum += (tau * qd).abs().sum(-1) * a
    self.vz_max = torch.maximum(self.vz_max, d.root_link_lin_vel_w[:, 2].abs() * a)
    self.wxy_max = torch.maximum(
      self.wxy_max, d.root_link_ang_vel_b[:, :2].norm(dim=-1) * a
    )
    margin = torch.minimum(q - t.q_min, t.q_max - q).amin(-1)
    self.margin_min = torch.where(
      act, torch.minimum(self.margin_min, margin), self.margin_min
    )
    sensor = env.scene["feet_contact"]
    f = sensor.data.force.norm(dim=-1).amax(-1)
    self.contact_max = torch.maximum(self.contact_max, f * a)
    self.samples += a

  def _q(self, bins: torch.Tensor, q: float, width: float) -> list[float]:
    cdf = torch.cumsum(bins, -1) / bins.sum(-1, keepdim=True).clamp(min=1)
    return (((cdf < q).sum(-1) + 1).float() * width).tolist()

  def group_summary(self, g: int, rows: torch.Tensor) -> dict:
    n = float(self.n[g].clamp(min=1))
    tau_all = self.tau_bins[g].sum(0)
    qd_all = self.qd_bins[g].sum(0)
    total = float(qd_all.sum().clamp(min=1))
    names = [x.replace("_joint", "") for x in JOINT_NAMES]
    s = {
      "tau_mean": dict(zip(names, (self.tau_sum[g] / n).tolist(), strict=True)),
      "tau_p95": dict(zip(names, self._q(self.tau_bins[g], 0.95, 0.01), strict=True)),
      "tau_p99": dict(zip(names, self._q(self.tau_bins[g], 0.99, 0.01), strict=True)),
      "tau_max": dict(zip(names, self.tau_max[g].tolist(), strict=True)),
      "qd_mean": dict(zip(names, (self.qd_sum[g] / n).tolist(), strict=True)),
      "qd_p95": dict(zip(names, self._q(self.qd_bins[g], 0.95, 0.01), strict=True)),
      "qd_p99": dict(zip(names, self._q(self.qd_bins[g], 0.99, 0.01), strict=True)),
      "qd_max": dict(zip(names, self.qd_max[g].tolist(), strict=True)),
      "tau_frac_near_cap": float(self.near_cap[g]) / (n * len(names)),
      "tau_frac_above_rated": float(self.above_rated[g]) / (n * len(names)),
      "tau_longest_above_rated_s": float(self.longest_rated[rows].max()),
      "qd_frac_above_3": float(qd_all[300:].sum()) / total,
      "qd_frac_above_4": float(qd_all[400:].sum()) / total,
      "qd_frac_above_6p28": float(qd_all[628:].sum()) / total,
      "episodes_with_qd_above_4": float((self.ep_qd_max[rows] > 4.0).float().mean()),
      "tracking_err_mean": float(
        (self.err_sum[rows] / self.samples[rows].clamp(min=1)).mean()
      ),
      "tracking_err_max": float(self.err_max[rows].max()),
      "mech_power_mean_W": float(
        (self.power_sum[rows] / self.samples[rows].clamp(min=1)).mean()
      ),
      "base_vz_max": float(self.vz_max[rows].max()),
      "base_wxy_max": float(self.wxy_max[rows].max()),
      "joint_limit_margin_min": float(self.margin_min[rows].min()),
      "foot_contact_force_max_N": float(self.contact_max[rows].max()),
    }
    del tau_all
    return s


def _attach(env, rec: Recorder):
  step = env.sim.step

  def hooked():
    step()
    rec()

  env.sim.step = hooked


def _height_summary(h: torch.Tensor, up: torch.Tensor, last: int) -> dict:
  """Base height / uprightness after the passive window (rows = envs of one pose).
  ``final`` = mean over the last ``last`` steps of each episode."""
  hf = h[:, -last:].mean(-1)
  uf = up[:, -last:].mean(-1)
  hmax = h.amax(-1)
  q = torch.tensor([0.1, 0.5, 0.9], device=h.device)
  return {
    "height_final_p10_p50_p90": hf.quantile(q).tolist(),
    "height_max_p10_p50_p90": hmax.quantile(q).tolist(),
    "frac_time_above_h1": float((h > STAGE_HEIGHTS[0]).float().mean()),
    "frac_time_above_h2": float((h > STAGE_HEIGHTS[1]).float().mean()),
    "frac_envs_final_above_h1": float((hf > STAGE_HEIGHTS[0]).float().mean()),
    "upright_cos_final_mean": float(uf.mean()),
    "frac_envs_final_upright": float((uf > STAND_UP).float().mean()),
  }


def _support_summary(sup: torch.Tensor, h: torch.Tensor, last: int) -> dict:
  """sup (n, T, groups) bool, h (n, T), actuated steps only. Support classes while
  the base is above h1: feet only / feet + shin (half kneel) / shin without feet
  (kneeling) / other. Also the class at the end of the episode (last 3 s)."""
  foot, calf = sup[..., 0], sup[..., 1]
  other = sup[..., 2:].any(-1)
  cls = torch.full(h.shape, 3, dtype=torch.long, device=h.device)
  cls[foot & ~calf & ~other] = 0
  cls[foot & calf & ~other] = 1
  cls[~foot & calf & ~other] = 2
  above = h > STAGE_HEIGHTS[0]
  names = ("feet_only", "feet_and_shin", "shin_no_feet", "other")
  n_above = above.sum().clamp(min=1)
  out = {
    f"above_h1_{nm}": float(((cls == i) & above).sum() / n_above)
    for i, nm in enumerate(names)
  }
  end = cls[:, -last:]
  end_above = above[:, -last:]
  for i, nm in enumerate(names):
    out[f"final_above_h1_{nm}"] = float(((end == i) & end_above).float().mean())
  out["final_below_h1"] = float((~end_above).float().mean())
  for i, g in enumerate(SUPPORT_GROUPS):
    out[f"touch_frac_{g}"] = float(sup[..., i].float().mean())
  return out


def evaluate_recovery(
  model, task, per_pose, policy_mode="student", assist_tc=None
) -> dict:
  npose = len(FALLEN_POSE_NAMES)
  n = per_pose * npose

  def mutate(cfg):
    cfg.events["reset_pose"].params["by_env_index"] = True

  env = build_env(task, n, mutate, assist_tc)
  t = env.action_manager.get_term("joint_pos")
  pose = env.extras[POSE_KEY].clone()
  support = SupportTracker(env)
  rec = Recorder(env, pose, npose, TAU_RATED_REPORT, TAU_CAP)
  _attach(env, rec)
  policy = StudentPolicy(model).to(DEV).eval()
  T = int(env.max_episode_length) - 1
  hist = torch.zeros(n, T, dtype=torch.bool, device=DEV)
  heights = torch.zeros(n, T, device=DEV)
  ups = torch.zeros(n, T, device=DEV)
  sup = torch.zeros(n, T, len(SUPPORT_GROUPS), dtype=torch.bool, device=DEV)
  t.monitor.clear()
  obs = env.get_observations()
  with torch.no_grad():
    for k in range(T):
      if policy_mode == "teacher":
        z = model.teacher_encoder(obs["teacher"])
        a = model.actor_mean(z, obs["actor"])
      else:
        a = policy(obs["policy"])
      obs, *_ = env.step(a)
      d = env.scene["robot"].data
      h = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
      up = -d.projected_gravity_b[:, 2]
      hist[:, k] = (h > STAND_H) & (up > STAND_UP)
      heights[:, k] = h
      ups[:, k] = up
      sup[:, k] = support.touching()
  dt = env.step_dt
  passive = t.cfg.passive_steps
  last = int(3.0 / dt)
  success = hist[:, -last:].all(-1)
  # Time to stand: first index starting a 1 s continuous standing run.
  w = int(1.0 / dt)
  runs = hist.float().unfold(1, w, 1).all(-1)
  first = torch.where(
    runs.any(-1), runs.float().argmax(-1), torch.full((n,), -1, device=DEV)
  )
  ever = runs.any(-1)
  # Fell after standing: standing run seen, then height below h2 later.
  fell = torch.zeros(n, dtype=torch.bool, device=DEV)
  for i in range(n):
    if bool(ever[i]):
      fell[i] = bool((heights[i, int(first[i]) :] < STAGE_HEIGHTS[1]).any())
  monitor = t.monitor.summary(t.joint_names)
  out = {
    "task": task,
    "plant": "PD16 (no envelope, no slew)",
    "policy": policy_mode,
    "assist_tc": assist_tc,
    "per_pose": {},
  }
  for g, name in enumerate(FALLEN_POSE_NAMES):
    rows = pose == g
    tts = (first[rows & ever].float() - passive) * dt
    out["per_pose"][name] = {
      "n": int(rows.sum()),
      "success": float(success[rows].float().mean()),
      "ever_stood": float(ever[rows].float().mean()),
      "fell_after_standing": float(fell[rows].float().mean()),
      "time_to_stand_s_mean": float(tts.mean()) if tts.numel() else None,
      "time_to_stand_s_p90": float(tts.quantile(0.9)) if tts.numel() else None,
      **_height_summary(heights[rows, passive:], ups[rows, passive:], last),
      **_support_summary(sup[rows, passive:], heights[rows, passive:], last),
      **rec.group_summary(g, rows),
    }
  out["success_min_pose"] = min(v["success"] for v in out["per_pose"].values())
  out["qd_frac_above_soft"] = monitor.get("qd_frac_above_soft", 0.0)
  env.close()
  return out


def evaluate_walk(model, task, per_cmd=16, episode_s=10.0) -> dict:
  vxs = (
    COMMAND_RANGES["lin_vel_x"][0],
    -0.15,
    0.0,
    0.15,
    0.3,
    COMMAND_RANGES["lin_vel_x"][1],
  )
  wzs = (COMMAND_RANGES["ang_vel_z"][0], 0.0, COMMAND_RANGES["ang_vel_z"][1])
  grid = (
    [(vx, 0.0) for vx in vxs]
    + [(0.2, wz) for wz in wzs if wz != 0.0]
    + [(0.0, wz) for wz in wzs if wz != 0.0]
  )
  n = per_cmd * len(grid)

  def mutate(cfg):
    cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    cfg.episode_length_s = episode_s + 1.0

  env = build_env(task, n, mutate)
  cmd_term = env.command_manager.get_term("twist")
  cmd = torch.tensor([grid[i // per_cmd] for i in range(n)], device=DEV)
  rec = Recorder(
    env, torch.zeros(n, dtype=torch.long, device=DEV), 1, TAU_RATED_REPORT, TAU_CAP
  )
  _attach(env, rec)
  policy = StudentPolicy(model).to(DEV).eval()
  sensor = env.scene["feet_contact"]
  T = int(episode_s / env.step_dt)
  settle = int(2.0 / env.step_dt)
  vx_sum = torch.zeros(n, device=DEV)
  wz_sum = torch.zeros(n, device=DEV)
  touchdowns = torch.zeros(n, 2, device=DEV)
  air_frac = torch.zeros(n, device=DEV)
  fell = torch.zeros(n, dtype=torch.bool, device=DEV)
  start_pos = None
  prev_contact = None

  def set_cmd():
    cmd_term.vel_command_b[:, 0] = cmd[:, 0]
    cmd_term.vel_command_b[:, 1] = 0.0
    cmd_term.vel_command_b[:, 2] = cmd[:, 1]

  set_cmd()
  obs = env.get_observations()
  with torch.no_grad():
    for k in range(T):
      set_cmd()
      obs, _, term_buf, _, _ = env.step(policy(obs["policy"]))
      set_cmd()
      fell |= term_buf
      d = env.scene["robot"].data
      contact = sensor.data.found > 0
      if k >= settle:
        vx_sum += d.root_link_lin_vel_b[:, 0]
        wz_sum += d.root_link_ang_vel_b[:, 2]
        if prev_contact is not None:
          touchdowns += (contact & ~prev_contact).float()
        air_frac += (~contact).float().mean(-1)
      if k == settle:
        start_pos = d.root_link_pos_w[:, :2].clone()
      prev_contact = contact
  steps = T - settle
  dur = steps * env.step_dt
  vx = vx_sum / steps
  wz = wz_sum / steps
  drift = (env.scene["robot"].data.root_link_pos_w[:, :2] - start_pos).norm(dim=-1)
  rows = []
  for i, (cvx, cwz) in enumerate(grid):
    sl = slice(i * per_cmd, (i + 1) * per_cmd)
    ok = ~fell[sl]
    rows.append(
      {
        "cmd_vx": cvx,
        "cmd_wz": cwz,
        "vx": float(vx[sl][ok].mean()) if ok.any() else None,
        "wz": float(wz[sl][ok].mean()) if ok.any() else None,
        "fall_rate": float(fell[sl].float().mean()),
        "touchdowns_per_s_per_foot": float(touchdowns[sl][ok].mean() / dur)
        if ok.any()
        else None,
        "air_fraction": float(air_frac[sl][ok].mean() / steps) if ok.any() else None,
        "drift_m": float(drift[sl][ok].mean()) if ok.any() else None,
      }
    )
  okm = ~fell
  corr_vx = float(torch.corrcoef(torch.stack((cmd[okm, 0], vx[okm])))[0, 1])
  corr_wz = float(torch.corrcoef(torch.stack((cmd[okm, 1], wz[okm])))[0, 1])
  out = {
    "task": task,
    "plant": "PD16 (no envelope, no slew)",
    "rows": rows,
    "corr_vx": corr_vx,
    "corr_wz": corr_wz,
    "fall_rate": float(fell.float().mean()),
    "physical": rec.group_summary(0, torch.ones(n, dtype=torch.bool, device=DEV)),
  }
  env.close()
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", required=True)
  ap.add_argument("--num-envs-per-pose", type=int, default=64)
  ap.add_argument("--policy", default="student", choices=("student", "teacher"))
  ap.add_argument("--out", default="")
  ap.add_argument(
    "--assist-tc",
    type=float,
    default=None,
    help="diagnostic: enable the training Eq. 4 assistance at this time coefficient",
  )
  args = ap.parse_args()
  model = load_model(args.checkpoint).to(DEV)
  if args.task.endswith("Walk"):
    out = evaluate_walk(model, args.task)
  else:
    out = evaluate_recovery(
      model,
      args.task,
      args.num_envs_per_pose,
      args.policy,
      args.assist_tc,
    )
  out["checkpoint"] = os.path.abspath(args.checkpoint)
  text = json.dumps(out, indent=1)
  if args.out:
    with open(args.out, "w") as f:
      f.write(text)
  print(text)


if __name__ == "__main__":
  main()
