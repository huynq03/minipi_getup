"""Deterministic FTSR checkpoint evaluation (simulation only; never touches hardware).

Recovery protocol (``--task recovery``, play cfg of ``Mjlab-FTSR-MiniPi``: no assist,
no observation noise, training domain randomization):

- env i starts in pose ``i % 4`` (supine, prone, left side, right side), perturbed
  as in training: roll/pitch +-0.3 rad, random yaw, joints +-0.3 rad. Velocity
  command ``(i // 4) % len(COMMANDS)``, fixed for the whole 20 s episode.
- One episode per env and seed. The deployable student policy (proprioceptive
  history only) is evaluated by default.
- Recovered: torso above 0.31 m and tilt below ~18 deg for 1 s in a row.
  ``time_to_stand`` is the start of that window, measured from the end of the settle.
  Success: recovered within 10 s and never fallen again (torso below 0.2 m or tilt
  above 45 deg) until the episode ends.
- Locomotion error is averaged from 2 s after recovery. Foot travel and base drift
  are measured over the same window, only for zero-command envs.

Walk protocol (``--task walk``, ``Mjlab-FTSR-MiniPi-Walk``): standing start, the same
commands, 10 s. It reports tracking error, upright share, falls, torque and smoothness.

Usage:
  python -m minipi_getup.ftsr.evaluate --checkpoints RUN/model_1000.pt ... \
      --seeds 0 1 2 --envs 400 --out logs/ftsr_analysis/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict

import torch

from minipi_getup.ftsr.mdp.actions import TorqueMonitor

POSES = ("supine", "prone", "left_side", "right_side")
# (vx, vy, wz): stand, forward, backward, turn, mixed.
COMMANDS = (
  (0.0, 0.0, 0.0),
  (0.3, 0.0, 0.0),
  (-0.2, 0.0, 0.0),
  (0.0, 0.0, 0.4),
  (0.2, 0.0, -0.3),
)
# ``--commands stand``: every env gets the zero command (get-up-first evaluation).
STAND_COMMANDS = ((0.0, 0.0, 0.0),)
STAND_HEIGHT = 0.31
STAND_UP = 0.95
FALL_HEIGHT = 0.2
FALL_UP = 0.707
HOLD_STEPS = 50


class _SplitMonitor:
  """Feeds torque samples to a recovery or a walking monitor by per-env phase."""

  def __init__(self, limit: float, device):
    self.recovery = TorqueMonitor(12, limit, device)
    self.walking = TorqueMonitor(12, limit, device)
    self.alive: torch.Tensor | None = None
    self.stood: torch.Tensor | None = None

  def add(self, tau, qvel):
    if self.alive is None or self.stood is None:
      return
    rec = self.alive & ~self.stood
    walk = self.alive & self.stood
    if rec.any():
      self.recovery.add(tau[rec], qvel[rec])
    if walk.any():
      self.walking.add(tau[walk], qvel[walk])

  def clear(self):
    self.recovery.clear()
    self.walking.clear()


def _build(
  task: str,
  num_envs: int,
  device: str,
  seed: int,
  action_relative: bool | None = None,
  action_scale: float | None = None,
  assist_iteration: int | None = None,
  env_task: str | None = None,
):
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from minipi_getup.ftsr.config.env_cfg import FALLEN_POSE_NAMES
  from minipi_getup.ftsr.mdp.events import FALLEN_POSES
  from minipi_getup.ftsr.rl import FtsrRunner

  task_id = "Mjlab-FTSR-MiniPi" if task == "recovery" else "Mjlab-FTSR-MiniPi-Walk"
  # Variants that change the robot or the action term (torque envelope, rate limit)
  # must be evaluated in their own env.
  task_id = env_task or task_id
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  assert "assist" not in cfg.events, "evaluation must run without assistance"
  if assist_iteration is not None:
    # Diagnostic only: keep the training-time Eq. 4 assist, frozen at the schedule
    # point of ``assist_iteration`` (the stage's h_cmd follows the stage manager).
    cfg.events["assist"] = load_env_cfg(task_id).events["assist"]
  if task == "recovery":
    assert FALLEN_POSE_NAMES == POSES
    cfg.events["reset_pose"].params["poses"] = tuple(FALLEN_POSES[p] for p in POSES)
    cfg.events["reset_pose"].params["by_env_index"] = True
  if action_relative is not None:
    cfg.actions["joint_pos"].relative = action_relative
  if action_scale is not None:
    cfg.actions["joint_pos"].scale = action_scale
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1e6, 1e6)
  twist.rel_standing_envs = 0.0
  env = ManagerBasedRlEnv(cfg, device=device)
  agent = load_rl_cfg(task_id)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = FtsrRunner(wrapped, asdict(agent), log_dir=None, device=device)
  return env, wrapped, runner


def _set_commands(env) -> torch.Tensor:
  term = env.command_manager.get_term("twist")
  idx = (torch.arange(env.num_envs, device=env.device) // len(POSES)) % len(COMMANDS)
  cmds = torch.tensor(COMMANDS, device=env.device)[idx]
  term.vel_command_b[:] = cmds
  term.is_standing_env[:] = False
  return cmds


@torch.no_grad()
def run_episode(env, wrapped, policy, seed: int, task: str, steps: int) -> dict:
  torch.manual_seed(seed)
  n, dev = env.num_envs, env.device
  obs, _ = wrapped.reset()
  cmds = _set_commands(env)
  # compute() returns the cached buffer unless forced; recompute with the new commands.
  env.observation_manager._obs_buffer = None
  obs = wrapped.get_observations()
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  split = _SplitMonitor(term.monitor.limit, dev)
  term.monitor = split
  dt = env.step_dt
  settle = term.cfg.settle_steps if task == "recovery" else 0

  alive = torch.ones(n, dtype=torch.bool, device=dev)
  stood = (
    torch.zeros(n, dtype=torch.bool, device=dev)
    if task == "recovery"
    else alive.clone()
  )
  run = torch.zeros(n, device=dev)
  t_stand = torch.full((n,), float("nan"), device=dev)
  fell_after = torch.zeros(n, dtype=torch.bool, device=dev)
  max_h = torch.zeros(n, device=dev)
  lin_err = torch.zeros(n, device=dev)
  yaw_err = torch.zeros(n, device=dev)
  track_n = torch.zeros(n, device=dev)
  tilt_sum = torch.zeros(n, device=dev)
  upright_n = torch.zeros(n, device=dev)
  steps_n = torch.zeros(n, device=dev)
  foot_travel = torch.zeros(n, device=dev)
  drift = torch.zeros(n, device=dev)
  drift_start = torch.full((n, 2), float("nan"), device=dev)
  feet_ids = robot.find_bodies(("r_ankle_roll_link", "l_ankle_roll_link"))[0]
  prev_feet = robot.data.body_link_pos_w[:, feet_ids, :2].clone()
  terminated = torch.zeros(n, dtype=torch.bool, device=dev)
  # Peak motion while getting up (until recovered): the "jerk upright" signature.
  peak_qd = torch.zeros(n, device=dev)
  peak_vz = torch.zeros(n, device=dev)
  peak_wxy = torch.zeros(n, device=dev)
  rate = smooth = qd_abs = qacc = 0.0
  motion_n = 0
  prev_qd = robot.data.joint_vel.clone()
  am = env.action_manager
  zero_cmd = cmds.abs().sum(-1) == 0

  for k in range(steps):
    split.alive, split.stood = alive.clone(), stood.clone()
    actions = policy(obs)
    obs, _, dones, _ = wrapped.step(actions)
    _set_commands(env)  # Keep the command fixed through any internal update.
    h = robot.data.body_link_pos_w[:, 0, 2]
    up = -robot.data.projected_gravity_b[:, 2]
    standing = (h > STAND_HEIGHT) & (up > STAND_UP)
    fallen = (h < FALL_HEIGHT) | (up < FALL_UP)
    active = alive & ~dones.bool()

    if task == "recovery":
      run = torch.where(standing & active, run + 1, torch.zeros_like(run))
      newly = (run >= HOLD_STEPS) & ~stood & active
      t_stand = torch.where(
        newly, (k + 1 - HOLD_STEPS - settle) * dt * torch.ones_like(t_stand), t_stand
      )
      stood |= newly
      fell_after |= stood & fallen & active
    else:
      fell_after |= fallen & active
    max_h = torch.where(alive, torch.maximum(max_h, h), max_h)
    # Peaks count from the end of the settle hold: the reset drop (~1 m/s torso v_z)
    # isn't the policy's motion.
    rec = active & ~stood & (k >= settle)
    qd_max = robot.data.joint_vel.abs().amax(dim=-1)
    vz = robot.data.root_link_lin_vel_w[:, 2].abs()
    wxy = robot.data.root_link_ang_vel_b[:, :2].norm(dim=-1)
    peak_qd = torch.where(rec, torch.maximum(peak_qd, qd_max), peak_qd)
    peak_vz = torch.where(rec, torch.maximum(peak_vz, vz), peak_vz)
    peak_wxy = torch.where(rec, torch.maximum(peak_wxy, wxy), peak_wxy)
    tilt = torch.acos(up.clamp(-1, 1))
    tilt_sum += torch.where(active & stood, tilt, torch.zeros_like(tilt))
    upright_n += (active & stood).float()
    steps_n += active.float()

    # Locomotion window: from 2 s after recovery.
    if task == "recovery":
      walk_win = active & stood & ((k + 1) * dt - settle * dt - t_stand > 2.0 + 1.0)
    else:
      walk_win = active & ~fell_after & (k * dt > 1.0)
    v = robot.data.root_link_lin_vel_b[:, :2]
    w = robot.data.root_link_ang_vel_b[:, 2]
    lin_err += torch.where(walk_win, (v - cmds[:, :2]).norm(dim=-1), 0.0)
    yaw_err += torch.where(walk_win, (w - cmds[:, 2]).abs(), 0.0)
    track_n += walk_win.float()
    feet = robot.data.body_link_pos_w[:, feet_ids, :2]
    step_travel = (feet - prev_feet).norm(dim=-1).sum(-1)
    still = walk_win & zero_cmd
    foot_travel += torch.where(still, step_travel, 0.0)
    base_xy = robot.data.root_link_pos_w[:, :2]
    start = still & torch.isnan(drift_start[:, 0])
    drift_start[start] = base_xy[start]
    drift = torch.where(still, (base_xy - drift_start).norm(dim=-1), drift)
    prev_feet = feet.clone()

    if active.any():
      a, a1, a2 = am.action[active], am.prev_action[active], am.prev_prev_action[active]
      rate += float(torch.sum((a - a1) ** 2, -1).mean())
      smooth += float(torch.sum((a - 2 * a1 + a2) ** 2, -1).mean())
      qd = robot.data.joint_vel
      qd_abs += float(qd[active].abs().mean())
      qacc += float((((qd - prev_qd) / dt)[active] ** 2).sum(-1).mean())
      motion_n += 1
    prev_qd = robot.data.joint_vel.clone()
    terminated |= dones.bool() & alive & (k < steps - 1)
    alive &= ~dones.bool()

  success = stood & ~fell_after
  if task == "recovery":
    success &= t_stand <= 10.0
  out = {
    "success": success.float(),
    "stood": stood.float(),
    "t_stand": t_stand,
    "fell_after": fell_after.float(),
    "terminated": terminated.float(),
    "max_h": max_h,
    "tilt_mean": tilt_sum / upright_n.clamp(min=1),
    "lin_err": lin_err / track_n.clamp(min=1),
    "yaw_err": yaw_err / track_n.clamp(min=1),
    "track_n": track_n,
    "foot_travel": foot_travel,
    "drift": drift,
    "zero_cmd": zero_cmd.float(),
    "peak_qd": peak_qd,
    "peak_vz": peak_vz,
    "peak_wxy": peak_wxy,
    "cmd_id": ((torch.arange(n, device=dev) // len(POSES)) % len(COMMANDS)).float(),
    "pose_id": (torch.arange(n, device=dev) % len(POSES)).float(),
  }
  motion = {
    "action_rate": rate / max(motion_n, 1),
    "action_smoothness": smooth / max(motion_n, 1),
    "joint_vel_abs": qd_abs / max(motion_n, 1),
    "joint_acc_sq": qacc / max(motion_n, 1),
  }
  torque = {
    "all": _merge(split.recovery, split.walking),
    "recovery": split.recovery.summary(),
    "walking": split.walking.summary(),
  }
  term.monitor = split.recovery  # Restore a plain monitor object.
  return {"per_env": out, "motion": motion, "torque": torque}


def _merge(a: TorqueMonitor, b: TorqueMonitor) -> dict:
  m = TorqueMonitor(12, a.limit, a.hist.device)
  m.hist = a.hist + b.hist
  m.count = a.count + b.count
  m.total = a.total + b.total
  m.saturated = a.saturated + b.saturated
  m.joint_max = torch.maximum(a.joint_max, b.joint_max)
  m.power_total = a.power_total + b.power_total
  m.power_count = a.power_count + b.power_count
  return m.summary()


def summarize(results: list[dict], task: str) -> dict:
  per = {
    k: torch.cat([r["per_env"][k] for r in results]) for k in results[0]["per_env"]
  }
  succ = per["success"]
  row: dict[str, float] = {
    "success_rate": float(succ.mean()),
    "stood_rate": float(per["stood"].mean()),
    "fell_after_rate": float(per["fell_after"].mean()),
    "termination_rate": float(per["terminated"].mean()),
    "max_base_height": float(per["max_h"].mean()),
    "orientation_error_deg": float(per["tilt_mean"][per["stood"] > 0].mean() * 57.3)
    if (per["stood"] > 0).any()
    else float("nan"),
  }
  if task == "recovery":
    ok = ~torch.isnan(per["t_stand"])
    row["mean_time_to_stand"] = (
      float(per["t_stand"][ok].mean()) if ok.any() else float("nan")
    )
    row["p90_time_to_stand"] = (
      float(per["t_stand"][ok].quantile(0.9)) if ok.any() else float("nan")
    )
    for i, p in enumerate(POSES):
      m = per["pose_id"] == i
      row[f"success_{p}"] = float(succ[m].mean())
  if task == "recovery":
    # Mean over episodes of the per-episode peak during the get-up, plus the worst.
    for k in ("peak_qd", "peak_vz", "peak_wxy"):
      row["recovery_" + k + "_mean"] = float(per[k].mean())
      row["recovery_" + k + "_max"] = float(per[k].max())
  tracked = per["track_n"] > 25
  moving = tracked & (per["zero_cmd"] == 0)
  row["lin_vel_error"] = (
    float(per["lin_err"][tracked].mean()) if tracked.any() else float("nan")
  )
  row["yaw_error"] = (
    float(per["yaw_err"][tracked].mean()) if tracked.any() else float("nan")
  )
  row["moving_lin_vel_error"] = (
    float(per["lin_err"][moving].mean()) if moving.any() else float("nan")
  )
  # Command-conditioned success: recovered, stayed up and tracked within 0.15 m/s,
  # 0.3 rad/s.
  good = (succ > 0) & tracked & (per["lin_err"] < 0.15) & (per["yaw_err"] < 0.3)
  row["command_success_rate"] = float(good.float().mean())
  still = tracked & (per["zero_cmd"] > 0)
  row["foot_travel"] = (
    float(per["foot_travel"][still].mean()) if still.any() else float("nan")
  )
  row["base_xy_drift"] = (
    float(per["drift"][still].mean()) if still.any() else float("nan")
  )
  for k in results[0]["motion"]:
    row[k] = sum(r["motion"][k] for r in results) / len(results)
  for phase in ("all", "recovery", "walking"):
    stats = [r["torque"][phase] for r in results if r["torque"][phase]]
    if not stats:
      continue
    prefix = "" if phase == "all" else phase + "_"
    for k in (
      "torque_mean",
      "torque_p95",
      "torque_p99",
      "torque_sat_frac",
      "mech_power_mean",
    ):
      row[prefix + k] = sum(s[k] for s in stats) / len(stats)
    row[prefix + "torque_max"] = max(s["torque_max"] for s in stats)
  return row


def append_rows(path: str, rows: list[dict]) -> None:
  """Append rows to a CSV, widening its header when new columns appear.

  Columns depend on the outcome (``walking_*`` only once some robot walks), so the
  file is rewritten with the union of all fields.
  """
  old: list[dict] = []
  fields: list[str] = []
  if os.path.exists(path):
    with open(path, newline="") as f:
      reader = csv.DictReader(f)
      fields = list(reader.fieldnames or [])
      old = list(reader)
  for row in rows:
    fields += [k for k in row if k not in fields]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(old + rows)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoints", nargs="+", required=True)
  ap.add_argument("--task", choices=("recovery", "walk"), default="recovery")
  ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  ap.add_argument("--envs", type=int, default=400)
  ap.add_argument("--seconds", type=float, default=None)
  ap.add_argument("--mode", choices=("student", "teacher"), default="student")
  ap.add_argument("--commands", choices=("mixed", "stand"), default="mixed")
  ap.add_argument(
    "--env-task",
    default=None,
    help="registered task whose play env to use (default: the faithful task)",
  )
  ap.add_argument("--run-name", default="")
  ap.add_argument("--out", default="logs/ftsr_analysis/results.csv")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--action-relative", choices=("true", "false"), default=None)
  ap.add_argument("--action-scale", type=float, default=None)
  ap.add_argument(
    "--assist-iteration",
    type=int,
    default=None,
    help="diagnostic: evaluate WITH the Eq. 4 assist at this training iteration",
  )
  args = ap.parse_args()
  if args.commands == "stand":
    global COMMANDS
    COMMANDS = STAND_COMMANDS

  seconds = args.seconds or (20.0 if args.task == "recovery" else 10.0)
  rel = None if args.action_relative is None else args.action_relative == "true"
  env, wrapped, runner = _build(
    args.task,
    args.envs,
    args.device,
    args.seeds[0],
    rel,
    args.action_scale,
    args.assist_iteration,
    args.env_task,
  )
  steps = int(round(seconds / env.step_dt))
  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  rows = []
  for ckpt in args.checkpoints:
    infos = runner.load(ckpt, map_location=args.device)
    env.common_step_counter = 24 * (args.assist_iteration or 0)
    policy = runner.get_inference_policy(device=args.device, mode=args.mode)
    results = [
      run_episode(env, wrapped, policy, s, args.task, steps) for s in args.seeds
    ]
    m = re.search(r"model_(\d+)\.pt", ckpt)
    row = {
      "run": args.run_name or os.path.basename(os.path.dirname(os.path.abspath(ckpt))),
      "checkpoint": ckpt,
      "iteration": int(m.group(1)) if m else -1,
      "task": args.task,
      "mode": args.mode,
      "seeds": " ".join(map(str, args.seeds)),
      "commands": args.commands,
      "env_task": args.env_task or "",
      "episodes": args.envs * len(args.seeds),
      "assistance_force": 0.0 if args.assist_iteration is None else -1.0,
      "assist_iteration": args.assist_iteration,
      "train_step_counter": (infos or {})
      .get("env_state", {})
      .get("common_step_counter"),
    }
    row.update(summarize(results, args.task))
    rows.append(row)
    print(
      json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
      )
    )
  append_rows(args.out, rows)
  print(f"wrote {len(rows)} rows to {args.out}")
  env.close()


if __name__ == "__main__":
  main()
