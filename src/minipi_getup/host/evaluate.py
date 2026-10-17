"""Evaluate a HoST-port checkpoint (HoST ``scripts/eval/eval_ground.py`` equivalent).

Release eval settings (``eval_ground.py``): deterministic policy, ``pull_force = False``,
``noise.add_noise = False``, ``control.action_scale = 0.25``, ``episode_length_s = 5``,
5 episodes, domain randomization as configured (on). ``--dr off`` disables every
randomization (the first evaluation the reproduction asks for).

Success, two definitions:

- ``host``: the release's G1 metric with Mini-Pi thresholds, evaluated from the first
  actuated step (Mini-Pi's 0.351 m reset height is above the scaled stand threshold). G1: stood = base height >
  0.7 m at some step, fell = afterwards base < 0.5 m, success = stood and not fell, with
  G1 ``base_height_target`` 0.75 m. Mini-Pi (``base_height_target`` 0.34 m): stand 0.317 m,
  fall 0.227 m.
- ``sustained``: base > 0.317 m *and* upright (projected gravity z < -0.8, the pull
  force's gate) for every step of the final 1.0 s of the episode. Rules out transient
  bounces that the ``host`` metric accepts.

uv run host-eval --checkpoint logs/host/Pi_ground/<run>/model_N.pt [--dr off|on]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from minipi_getup.host.config import PiCfg, PiCfgPPO, class_to_dict

DEV = "cuda:0"


def load_policy(path: str, env) -> torch.nn.Module:
  from minipi_getup.host_rl.actor_critic import ActorCritic

  pcfg = class_to_dict(PiCfgPPO())["policy"]
  ac = ActorCritic(
    env.num_obs, env.num_obs, env.num_actions, env.cfg.rewards.num_reward_groups, **pcfg
  ).to(DEV)
  ac.load_state_dict(torch.load(path, map_location=DEV)["model_state_dict"])
  ac.eval()
  return ac


def make_eval_cfg(dr: bool, pull_force: bool, action_scale: float, episode_s: float):
  cfg = PiCfg()
  cfg.noise.add_noise = False
  cfg.env.episode_length_s = episode_s
  cfg.control.action_scale = action_scale
  cfg.curriculum.pull_force = pull_force
  cfg.env.test = True
  if not dr:
    d = cfg.domain_rand
    for k in dir(d):
      if k.startswith("randomize_") or k == "delay":
        setattr(d, k, False)
  return cfg


@torch.inference_mode()
def rollout(
  checkpoint: str,
  num_envs: int = 256,
  episodes: int = 5,
  dr: bool = False,
  pull_force: bool = False,
  action_scale: float = 0.25,
  episode_s: float = 5.0,
  record_envs: tuple[int, ...] = (),
  seed: int = 0,
):
  from minipi_getup.host.env import FALL_HEIGHT, STAND_HEIGHT, LeggedRobot_Pi

  torch.manual_seed(seed)
  np.random.seed(seed)
  cfg = make_eval_cfg(dr, pull_force, action_scale, episode_s)
  env = LeggedRobot_Pi(cfg, num_envs, DEV)
  policy = load_policy(checkpoint, env)
  obs, _ = env.reset()
  T = int(env.max_episode_length)
  dt = env.dt
  t_act = env.unactuated_time * dt
  hold = int(round(1.0 / dt))
  out = {
    k: []
    for k in (
      "host",
      "sustained",
      "time_to_stand",
      "peak_torque",
      "peak_qd",
      "mean_qd",
      "energy",
      "max_base",
      "final_base",
      "final_up",
    )
  }
  per_joint_peak = torch.zeros(env.num_dof, device=DEV)
  traces = {i: [] for i in record_envs}
  for _ep in range(episodes):
    stood = torch.zeros(num_envs, dtype=torch.bool, device=DEV)
    fell = torch.zeros_like(stood)
    t_stand = torch.full((num_envs,), float("nan"), device=DEV)
    peak_tq = torch.zeros(num_envs, env.num_dof, device=DEV)
    peak_qd = torch.zeros(num_envs, device=DEV)
    sum_qd = torch.zeros(num_envs, device=DEV)
    energy = torch.zeros(num_envs, device=DEV)
    max_base = torch.zeros(num_envs, device=DEV)
    alive = torch.ones(num_envs, dtype=torch.bool, device=DEV)
    hold_buf = torch.zeros(hold, num_envs, dtype=torch.bool, device=DEV)
    final_base = torch.zeros(num_envs, device=DEV)
    final_up = torch.zeros(num_envs, device=DEV)
    n_steps = torch.zeros(num_envs, device=DEV)
    for _j in range(T + 2):
      actions = policy.act_inference(obs)
      obs, _, _, _, _ = env.step(actions)
      # An env whose episode ended in this step is already reset (the release's
      # reset_idx runs inside step): freeze its metrics at the previous step.
      alive &= ~env.reset_buf.bool()
      if not alive.any():
        break
      h = env.root_states[:, 2]
      pgz = env.projected_gravity[:, 2]
      up = pgz < -0.8
      s_now = (h > STAND_HEIGHT) & up
      # Mini-Pi resets at 0.351 m > STAND_HEIGHT (G1: 0.5 m < 0.7 m), so the drop
      # during the unactuated window must not count as standing.
      actuated = env.real_episode_length_buf > env.unactuated_time
      t = env.real_episode_length_buf.float() * dt - t_act  # s since actuation began
      t_stand = torch.where(alive & s_now & torch.isnan(t_stand), t, t_stand)
      stood |= (h > STAND_HEIGHT) & actuated & alive
      fell |= stood & (h < FALL_HEIGHT) & alive
      hold_buf = torch.where(alive, torch.cat((hold_buf[1:], s_now[None])), hold_buf)
      peak_tq = torch.where(
        alive[:, None], torch.maximum(peak_tq, env.peak_torque_step), peak_tq
      )
      qd = env.dof_vel.abs()
      peak_qd = torch.where(alive, torch.maximum(peak_qd, qd.max(-1).values), peak_qd)
      sum_qd += qd.mean(-1) * alive
      n_steps += alive.float()
      energy += env.step_energy * alive
      max_base = torch.where(alive & actuated, torch.maximum(max_base, h), max_base)
      final_base = torch.where(alive, h, final_base)
      final_up = torch.where(alive, pgz, final_up)
      for i in record_envs:
        if not bool(alive[i]):
          continue
        traces[i].append(
          {
            "qpos": env.sim.data.qpos[i].cpu().numpy().copy(),
            "t": float(t[i]) + t_act,
            "base_z": float(h[i]),
            "head": float(env.old_headheight[i]),
            "pg_z": float(pgz[i]),
            "force": float(env.pending_force[i, 0, 2]),
            "action_rescale": float(env.action_rescale[i]),
            "max_torque": float(env.torques[i].abs().max()),
          }
        )
    ok_hold = hold_buf.all(0)
    # Restart the remaining envs so the next episode starts synchronized.
    env.reset_idx(torch.arange(num_envs, device=DEV))
    obs, *_ = env.step(torch.zeros(num_envs, env.num_actions, device=DEV))
    out["host"].append((stood & ~fell).float())
    out["sustained"].append(ok_hold.float())
    out["time_to_stand"].append(t_stand)
    out["peak_torque"].append(peak_tq.max(-1).values)
    out["peak_qd"].append(peak_qd)
    out["mean_qd"].append(sum_qd / n_steps.clamp(min=1))
    out["energy"].append(energy)
    out["max_base"].append(max_base)
    out["final_base"].append(final_base)
    out["final_up"].append(final_up)
    per_joint_peak = torch.maximum(per_joint_peak, peak_tq.max(0).values)
  cat = {k: torch.cat(v) for k, v in out.items()}
  tts = cat["time_to_stand"]
  summary = {
    "checkpoint": checkpoint,
    "settings": {
      "num_envs": num_envs,
      "episodes": episodes,
      "episode_s": episode_s,
      "domain_randomization": dr,
      "pull_force": pull_force,
      "action_scale": action_scale,
      "policy": "deterministic (act_inference)",
      "obs_noise": False,
    },
    "success_host_metric": float(cat["host"].mean()),
    "success_sustained_upright_last_1s": float(cat["sustained"].mean()),
    "success_host_std_over_episodes": float(torch.stack(out["host"]).mean(1).std()),
    "time_to_stand_s_mean": float(tts[~torch.isnan(tts)].mean())
    if (~torch.isnan(tts)).any()
    else None,
    "time_to_stand_s_median": float(tts[~torch.isnan(tts)].median())
    if (~torch.isnan(tts)).any()
    else None,
    "stood_upright_frac": float((~torch.isnan(tts)).float().mean()),
    "peak_torque_mean": float(cat["peak_torque"].mean()),
    "peak_torque_max": float(cat["peak_torque"].max()),
    "per_joint_peak_torque": dict(
      zip(env.dof_names, per_joint_peak.tolist(), strict=True)
    ),
    "joint_vel_mean": float(cat["mean_qd"].mean()),
    "joint_vel_peak_mean": float(cat["peak_qd"].mean()),
    "joint_vel_peak_max": float(cat["peak_qd"].max()),
    "energy_J_mean": float(cat["energy"].mean()),
    "max_base_height_mean": float(cat["max_base"].mean()),
    "final_base_height_mean": float(cat["final_base"].mean()),
    "final_proj_gravity_z_mean": float(cat["final_up"].mean()),
    "thresholds": {
      "stand_base_m": STAND_HEIGHT,
      "fall_base_m": FALL_HEIGHT,
      "upright_pg_z": -0.8,
    },
  }
  return summary, traces, env


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--episodes", type=int, default=5)
  ap.add_argument("--dr", choices=["off", "on"], default="off")
  ap.add_argument("--pull-force", choices=["off", "on"], default="off")
  ap.add_argument("--action-scale", type=float, default=0.25)
  ap.add_argument("--episode-s", type=float, default=5.0)
  ap.add_argument("--out", default=None, help="json path")
  args = ap.parse_args()
  summary, _, _ = rollout(
    args.checkpoint,
    args.num_envs,
    args.episodes,
    dr=args.dr == "on",
    pull_force=args.pull_force == "on",
    action_scale=args.action_scale,
    episode_s=args.episode_s,
  )
  print(json.dumps(summary, indent=2))
  if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
      json.dump(summary, f, indent=2)


if __name__ == "__main__":
  main()
