"""Unassisted, orientation-resolved evaluation of staged or safe-recovery checkpoints.

Both variants are measured with the same definitions: same seed and reset
distribution, DR and observation noise off, pull force off, one fixed action scale.
Optionally saves joint/root trajectories so videos can be rendered without
re-simulating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from minipi_getup.host.config import class_to_dict
from minipi_getup.host.env import FALL_HEIGHT, STAND_HEIGHT
from minipi_getup.host_rl.actor_critic import ActorCritic
from minipi_getup.host_safe_recovery.env import RESET_CLASSES, _saturating_support
from minipi_getup.host_staged_recovery import geometry as geo
from minipi_getup.host_staged_recovery.config import StagedRecoveryCfg

STAGE = StagedRecoveryCfg.staged


def build(variant: str, action_scale: float, episode_s: float, num_envs: int, device: str):
  if variant == "staged":
    from minipi_getup.host_staged_recovery import EXPERIMENT_NAME as name
    from minipi_getup.host_staged_recovery.asset import asset_sha256, configure_asset
    from minipi_getup.host_staged_recovery.config import StagedRecoveryPPOCfg as Ppo
    from minipi_getup.host_staged_recovery.env import StagedRecoveryEnv as Env
    cfg = StagedRecoveryCfg()
  else:
    from minipi_getup.host_safe_recovery import EXPERIMENT_NAME as name
    from minipi_getup.host_safe_recovery.asset import asset_sha256, configure_asset
    from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg
    from minipi_getup.host_safe_recovery.config import SafeRecoveryPPOCfg as Ppo
    from minipi_getup.host_safe_recovery.env import SafeRecoveryEnv as Env
    cfg = SafeRecoveryCfg()
  configure_asset()
  cfg.noise.add_noise = False
  cfg.env.episode_length_s = episode_s
  cfg.control.action_scale = action_scale
  cfg.curriculum.pull_force = False
  cfg.env.test = True
  for key in dir(cfg.domain_rand):
    if key.startswith("randomize_") or key == "delay":
      setattr(cfg.domain_rand, key, False)
  env = Env(cfg, num_envs, device)
  return env, name, asset_sha256(), Ppo


def load_actor(checkpoint: Path, env, name: str, sha: str, ppo_cls, device: str) -> ActorCritic:
  saved = torch.load(checkpoint, map_location=device, weights_only=False)
  info = saved.get("infos") or {}
  if info.get("variant") != name or info.get("asset_sha256") != sha:
    raise ValueError(f"{checkpoint}: variant/asset {info.get('variant')}/{info.get('asset_sha256')} != {name}/{sha}")
  actor = ActorCritic(
    env.num_obs, env.num_obs, env.num_actions, env.cfg.rewards.num_reward_groups,
    **class_to_dict(ppo_cls())["policy"],
  ).to(device)
  actor.load_state_dict(saved["model_state_dict"])
  actor.eval()
  return actor


def _stats(x: torch.Tensor) -> dict:
  x = x[torch.isfinite(x)].float()
  if not len(x):
    return {"mean": None, "p50": None, "p95": None, "max": None, "n": 0}
  return {"mean": float(x.mean()), "p50": float(x.median()), "p95": float(torch.quantile(x, .95)),
          "max": float(x.max()), "n": int(len(x))}


@torch.inference_mode()
def rollout(args) -> dict:
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  dev = args.device
  env, name, sha, ppo = build(args.variant, args.action_scale, args.episode_s, args.num_envs, dev)
  actor = load_actor(args.checkpoint, env, name, sha, ppo, dev)
  n = env.num_envs
  weight = env.sim.model.body_mass[:, env.robot.indexing.body_ids.long()].sum(1) * 9.81
  sr = env.cfg.safe_recovery
  calf_idx = env.knee_contact_indices
  hold_steps = round(1.0 / env.dt)

  sub = {"qd": torch.zeros(n, env.num_dof, device=dev), "tau": torch.zeros(n, env.num_dof, device=dev)}
  orig = env._substep_monitor

  def monitor():
    orig()
    act = (env.real_episode_length_buf > env.unactuated_time)[:, None]
    sub["qd"] = torch.maximum(sub["qd"], env.dof_vel.abs() * act)
    sub["tau"] = torch.maximum(sub["tau"], env.torques.abs() * act)
  env._substep_monitor = monitor

  obs, _ = env.reset()
  records, trajs = [], []
  for _ in range(args.episodes):
    cls = env.reset_class.clone()
    sub["qd"] = torch.zeros(n, env.num_dof, device=dev)
    sub["tau"] = torch.zeros(n, env.num_dof, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    stood = torch.zeros_like(alive)
    fell = torch.zeros_like(alive)
    t_stand, t_feet, t_com, t_rise = (torch.full((n,), float("nan"), device=dev) for _ in range(4))
    hold = torch.zeros(hold_steps, n, dtype=torch.bool, device=dev)
    knee_time = torch.zeros(n, device=dev)
    com_err_rise, com_err_rise_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    final = {k: torch.zeros(n, device=dev) for k in ("pose_rms", "calf_min", "width", "gap", "com_err", "tilt_deg", "qd_rms")}
    rise_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    rise_width = torch.full((n,), float("nan"), device=dev)
    rise_com = torch.full((n,), float("nan"), device=dev)
    prev_h = torch.zeros(n, device=dev)
    traj = []
    memory = torch.zeros(n, 2, device=dev)
    for _step in range(int(env.max_episode_length) + 2):
      obs, *_ = env.step(actor.act_inference(obs))
      ended = env.reset_buf.bool()
      active = alive & ~ended
      d = env.robot.data
      h = env.root_states[:, 2]
      upright = -env.projected_gravity[:, 2]
      actuated = env.real_episode_length_buf > env.unactuated_time
      t = (env.real_episode_length_buf.float() - env.unactuated_time) * env.dt
      standing = (h > STAND_HEIGHT) & (upright > 0.8)
      stood |= active & actuated & standing
      fell |= active & stood & (h < FALL_HEIGHT)
      hold = torch.where(active, torch.cat((hold[1:], standing[None])), hold)
      t_stand = torch.where(active & standing & torch.isnan(t_stand) & actuated, t, t_stand)
      # Geometry, identical for both variants.
      com = d.data.subtree_com[:, env.robot.indexing.root_body_id]
      fpos, fquat = d.body_link_pos_w[:, env.sole_body_indices], d.body_link_quat_w[:, env.sole_body_indices]
      corners = geo.sole_corners_world(fpos, fquat)
      left, right, _ = env._normalized_supports()
      load = torch.stack((left, right), 1)
      memory = torch.maximum(load, memory * 0.8)  # simple sustained-support proxy
      valid = (corners[..., 2] < STAGE.corner_contact_height_m) & (load[:, :, None] > STAGE.support_foot_min)
      outside, centre, anyv = geo.support_polygon_distance(com[:, :2], corners[..., :2].reshape(-1, 8, 2), valid.reshape(-1, 8))
      score = geo.com_support_score(outside, centre, anyv, STAGE.com_support_sigma_m, STAGE.com_centre_sigma_m)
      lift = h > STAGE.lift_gate_height_center
      sustained = torch.minimum(load, memory).min(1).values
      cen = corners.mean(2)
      width = torch.linalg.vector_norm(cen[:, 0, :2] - cen[:, 1, :2], dim=-1)
      gap = geo.foot_lateral_gap(corners, fquat)
      feet = sustained > STAGE.stage_feet_support
      supported = feet & geo.adequate_stance(gap, width, STAGE.stance_min_gap_m, STAGE.stance_max_width_m) & (
        outside <= STAGE.stage_com_supported_m)
      for buf, hit in ((t_feet, feet), (t_com, supported & (score > STAGE.stage_com_score) & lift)):
        buf.copy_(torch.where(active & actuated & hit & torch.isnan(buf), t, buf))
      crossing = active & actuated & ~stood & (h > STAGE.stage_rise_height_m) & (prev_h <= STAGE.stage_rise_height_m) & (
        upright > STAGE.stage_rise_upright)
      t_rise = torch.where(crossing, t, t_rise)
      rise_ok = torch.where(crossing, supported, rise_ok)
      rise_width = torch.where(crossing, width, rise_width)
      rise_com = torch.where(crossing, torch.where(anyv, outside, torch.full_like(outside, float("nan"))), rise_com)
      prev_h = h.clone()
      rising = active & actuated & lift & ~standing & anyv
      com_err_rise += torch.where(rising, outside, torch.zeros_like(outside))
      com_err_rise_n += rising
      history = env.nonfoot_ground_sensor.data.force_history
      knee = history[:, env.knee_force_indices, :, 2].abs().mean(dim=2).sum(dim=1)
      knee_sup = _saturating_support(knee, weight, sr.contact_force_deadband_weight_fraction, sr.contact_force_scale_weight_fraction)
      knee_time += (active & actuated & (knee_sup > STAGE.knee_contact_support_min)).float() * env.dt
      tilt = torch.stack([geo_tilt(q) for q in fquat.unbind(1)], 1).max(1).values
      vals = {
        "pose_rms": torch.sqrt(((env.dof_pos - env.default_dof_pos) ** 2).mean(1)),
        "calf_min": env.dof_pos[:, calf_idx].min(1).values,
        "width": width,
        "gap": gap,
        "com_err": torch.where(anyv, outside, torch.full_like(outside, float("nan"))),
        "tilt_deg": tilt,
        "qd_rms": torch.sqrt((env.dof_vel ** 2).mean(1)),
      }
      for k, v in vals.items():
        final[k] = torch.where(active, v, final[k])
      if args.save_traj and len(traj) < env.max_episode_length:
        traj.append(torch.cat((env.root_states[:, :7], env.dof_pos), 1)[: args.save_traj].cpu())
      alive = active
      if not alive.any():
        break
    sustained_end = hold.all(0)
    success = stood & ~fell & sustained_end
    records.append({
      "class": cls, "success": success, "stood": stood, "t_stand": t_stand,
      "t_feet": t_feet, "t_com": t_com, "t_rise": t_rise,
      "peak_qd": sub["qd"].max(1).values, "peak_qd_joint": sub["qd"], "peak_tau_joint": sub["tau"],
      "knee_time": knee_time, "com_err_rise": com_err_rise / com_err_rise_n.clamp_min(1),
      "rise_ok": rise_ok, "rise_width": rise_width, "rise_com": rise_com,
      **{f"final_{k}": v for k, v in final.items()},
    })
    if args.save_traj:
      trajs.append((cls[: args.save_traj].cpu(), torch.stack(traj, 1)))
    env.reset_idx(torch.arange(n, device=dev))
    obs, *_ = env.step(torch.zeros(n, env.num_actions, device=dev))

  cat = {k: torch.cat([r[k] for r in records]) for k in records[0]}
  ceiling = 6.0

  def summarize(m):
    s = cat["success"][m]
    ok = m & cat["stood"]
    order = lambda a, b: ((cat[a] <= cat[b]) & torch.isfinite(cat[a]))[ok].float().mean()  # noqa: E731
    return {
      "episodes": int(m.sum()),
      "success": float(s.float().mean()),
      "time_to_stand_s": _stats(cat["t_stand"][m & cat["success"]]),
      "peak_joint_velocity_rad_s": _stats(cat["peak_qd"][m]),
      "velocity_violation_frac": float((cat["peak_qd"][m] > ceiling).float().mean()),
      "peak_torque_nm": float(cat["peak_tau_joint"][m].max()) if m.any() else None,
      "final_stance_width_m": _stats(cat["final_width"][m & cat["success"]]),
      "final_foot_gap_m": _stats(cat["final_gap"][m & cat["success"]]),
      "final_com_support_error_m": _stats(cat["final_com_err"][m & cat["success"]]),
      "supported_at_rise_onset": float(cat["rise_ok"][ok].float().mean()) if ok.any() else None,
      "stance_width_at_rise_onset_m": _stats(cat["rise_width"][ok]),
      "com_error_at_rise_onset_m": _stats(cat["rise_com"][ok]),
      "rise_com_support_error_m": _stats(cat["com_err_rise"][m]),
      "final_posture_rms_rad": _stats(cat["final_pose_rms"][m & cat["success"]]),
      "final_reverse_knee_frac": float((cat["final_calf_min"][m & cat["success"]] < -STAGE.knee_reverse_soft_rad).float().mean()),
      "final_max_sole_tilt_deg": _stats(cat["final_tilt_deg"][m & cat["success"]]),
      "knee_contact_time_s": _stats(cat["knee_time"][m]),
      "stage_feet_before_rise": float(order("t_feet", "t_rise")) if ok.any() else None,
      "stage_com_before_rise": float(order("t_com", "t_rise")) if ok.any() else None,
      "stage_t_feet_s": _stats(cat["t_feet"][ok]),
      "stage_t_com_s": _stats(cat["t_com"][ok]),
      "stage_t_rise_s": _stats(cat["t_rise"][ok]),
    }

  out = {
    "checkpoint": str(args.checkpoint.resolve()), "variant": args.variant, "asset_sha256": sha,
    "settings": {"num_envs": n, "episodes": args.episodes, "seed": args.seed, "dr": False,
                 "pull_force": False, "observation_noise": False, "action_scale": args.action_scale,
                 "episode_s": args.episode_s, "velocity_ceiling_rad_s": ceiling},
    "overall": summarize(torch.ones_like(cat["success"])),
    "per_orientation": {c: summarize(cat["class"] == i) for i, c in enumerate(RESET_CLASSES)},
    "per_joint_peak_velocity_rad_s": dict(zip(env.dof_names, cat["peak_qd_joint"].max(0).values.tolist(), strict=True)),
    "per_joint_peak_torque_nm": dict(zip(env.dof_names, cat["peak_tau_joint"].max(0).values.tolist(), strict=True)),
  }
  if args.save_traj:
    np.savez_compressed(
      args.out.with_suffix(".traj.npz"),
      classes=torch.cat([c for c, _ in trajs]).numpy(),
      qpos=torch.cat([q for _, q in trajs]).numpy(), dt=env.dt,
      dof_names=np.array(env.dof_names), reset_classes=np.array(RESET_CLASSES),
    )
  return out


def geo_tilt(quat: torch.Tensor) -> torch.Tensor:
  """Sole-plane tilt against the ground, degrees."""
  down = torch.zeros(len(quat), 3, device=quat.device)
  down[:, 2] = -1.0
  g = quat_apply_inverse(quat, down)
  return torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1.0, 1.0)))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--variant", choices=("staged", "safe"), default="staged")
  p.add_argument("--num-envs", type=int, default=250)
  p.add_argument("--episodes", type=int, default=1)
  p.add_argument("--action-scale", type=float, default=0.25)
  p.add_argument("--episode-s", type=float, default=10.0)
  p.add_argument("--seed", type=int, default=20261012)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--save-traj", type=int, default=0, help="save qpos of the first N envs")
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()
  result = rollout(args)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2))
  o = result["overall"]
  print(f"{args.variant} {args.checkpoint.name} scale={args.action_scale}: success {o['success']:.3f}, "
        f"peak qd p95 {o['peak_joint_velocity_rad_s']['p95']}, >6 rad/s {o['velocity_violation_frac']:.2f}, "
        f"supported-at-rise {o['supported_at_rise_onset']}, com-before-rise {o['stage_com_before_rise']}, "
        f"feet-before-rise {o['stage_feet_before_rise']}")
  for c, v in result["per_orientation"].items():
    print(f"  {c:12s} n={v['episodes']:4d} success={v['success']:.3f} "
          f"width={v['final_stance_width_m']['p50']} gap={v['final_foot_gap_m']['p50']} "
          f"rise_width={v['stance_width_at_rise_onset_m']['p50']} supp@rise={v['supported_at_rise_onset']} com_err={v['final_com_support_error_m']['p50']} "
          f"reverse_knee={v['final_reverse_knee_frac']:.2f} knee_s={v['knee_contact_time_s']['mean']}")


if __name__ == "__main__":
  main()
