"""Deterministic, orientation-resolved evaluation for host_safe_recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minipi_getup.host.config import class_to_dict
from minipi_getup.host.env import FALL_HEIGHT, STAND_HEIGHT
from minipi_getup.host_rl.actor_critic import ActorCritic
from minipi_getup.host_safe_recovery import EXPERIMENT_NAME
from minipi_getup.host_safe_recovery.asset import (
  asset_sha256,
  configure_asset,
)
from minipi_getup.host_safe_recovery.config import SafeRecoveryCfg, SafeRecoveryPPOCfg
from minipi_getup.host_safe_recovery.env import RESET_CLASSES, SafeRecoveryEnv


def make_cfg(dr: bool, action_scale: float, episode_s: float) -> SafeRecoveryCfg:
  cfg = SafeRecoveryCfg()
  cfg.noise.add_noise = False
  cfg.env.episode_length_s = episode_s
  cfg.control.action_scale = action_scale
  cfg.curriculum.pull_force = False
  cfg.env.test = True
  if not dr:
    for key in dir(cfg.domain_rand):
      if key.startswith("randomize_") or key == "delay":
        setattr(cfg.domain_rand, key, False)
  return cfg


def load_policy(checkpoint: Path, env: SafeRecoveryEnv, device: str) -> ActorCritic:
  saved = torch.load(checkpoint, map_location=device, weights_only=False)
  info = saved.get("infos") or {}
  if info.get("variant") != EXPERIMENT_NAME:
    raise ValueError(f"not a {EXPERIMENT_NAME} checkpoint: {checkpoint}")
  if info.get("asset_sha256") != asset_sha256():
    raise ValueError("checkpoint asset hash does not match the required XML")
  policy_cfg = class_to_dict(SafeRecoveryPPOCfg())["policy"]
  actor = ActorCritic(
    env.num_obs,
    env.num_obs,
    env.num_actions,
    env.cfg.rewards.num_reward_groups,
    **policy_cfg,
  ).to(device)
  actor.load_state_dict(saved["model_state_dict"])
  actor.eval()
  return actor


def _stats(values: torch.Tensor) -> dict[str, float | None]:
  finite = values[torch.isfinite(values)]
  if not len(finite):
    return {"mean": None, "median": None, "p95": None, "max": None}
  return {
    "mean": float(finite.mean()),
    "median": float(finite.median()),
    "p95": float(torch.quantile(finite, 0.95)),
    "max": float(finite.max()),
  }


@torch.inference_mode()
def rollout(
  checkpoint: Path,
  num_envs: int,
  episodes: int,
  dr: bool,
  action_scale: float,
  episode_s: float,
  seed: int,
  device: str,
) -> dict:
  torch.manual_seed(seed)
  np.random.seed(seed)
  configure_asset()
  cfg = make_cfg(dr, action_scale, episode_s)
  env = SafeRecoveryEnv(cfg, num_envs, device)
  actor = load_policy(checkpoint, env, device)
  obs, _ = env.reset()
  horizon = int(env.max_episode_length) + 2
  hold_steps = round(1.0 / env.dt)
  records: list[dict[str, torch.Tensor]] = []

  for _ in range(episodes):
    reset_class = env.reset_class.clone()
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    stood = torch.zeros_like(alive)
    fell = torch.zeros_like(alive)
    time_to_stand = torch.full((num_envs,), float("nan"), device=device)
    peak_qd_joint = torch.zeros(num_envs, env.num_dof, device=device)
    overspeed_count = torch.zeros_like(peak_qd_joint)
    peak_torque_joint = torch.zeros_like(peak_qd_joint)
    saturation_count = torch.zeros_like(peak_qd_joint)
    sample_count = torch.zeros(num_envs, device=device)
    rom_violation = torch.zeros_like(alive)
    hold = torch.zeros(hold_steps, num_envs, dtype=torch.bool, device=device)
    final_pose_rms = torch.zeros(num_envs, device=device)
    final_qd_rms = torch.zeros(num_envs, device=device)
    final_base_ang = torch.zeros(num_envs, device=device)
    final_tilt = torch.zeros(num_envs, device=device)

    for _step in range(horizon):
      obs, _, _, _, _ = env.step(actor.act_inference(obs))
      ended = env.reset_buf.bool()
      active = alive & ~ended
      h = env.root_states[:, 2]
      upright = env.projected_gravity[:, 2] < -0.8
      standing = (h > STAND_HEIGHT) & upright
      actuated = env.real_episode_length_buf > env.unactuated_time
      t = (env.real_episode_length_buf.float() - env.unactuated_time) * env.dt
      time_to_stand = torch.where(
        active & standing & torch.isnan(time_to_stand), t, time_to_stand
      )
      stood |= active & actuated & (h > STAND_HEIGHT)
      fell |= active & stood & (h < FALL_HEIGHT)
      hold = torch.where(active, torch.cat((hold[1:], standing[None])), hold)

      # Environment monitors sample each 5 ms control substep, not just policy steps.
      peak_qd_joint = torch.where(
        active[:, None], torch.maximum(peak_qd_joint, env.peak_joint_vel), peak_qd_joint
      )
      overspeed_count = torch.where(
        active[:, None], env.overspeed_steps_by_joint, overspeed_count
      )
      torque = env.torques.abs()
      peak_torque_joint = torch.where(
        active[:, None], torch.maximum(peak_torque_joint, torque), peak_torque_joint
      )
      saturation_count += active[:, None] * (
        torque >= env.torque_limits.unsqueeze(0) - 1e-5
      )
      sample_count += active
      lower = env.robot.data.joint_pos_limits[0, :, 0]
      upper = env.robot.data.joint_pos_limits[0, :, 1]
      rom_violation |= active & (
        ((env.dof_pos < lower) | (env.dof_pos > upper)).any(1)
      )
      final_pose_rms = torch.where(
        active,
        torch.sqrt(torch.mean((env.dof_pos - env.default_dof_pos).square(), 1)),
        final_pose_rms,
      )
      final_qd_rms = torch.where(
        active, torch.sqrt(torch.mean(env.dof_vel.square(), 1)), final_qd_rms
      )
      final_base_ang = torch.where(
        active, torch.linalg.vector_norm(env.base_ang_vel, dim=1), final_base_ang
      )
      final_tilt = torch.where(
        active,
        torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1, 1))),
        final_tilt,
      )
      alive = active
      if not alive.any():
        break

    sustained = hold.all(0)
    velocity_safe = peak_qd_joint.max(1).values <= cfg.safe_recovery.evaluation_ceiling_rad_s
    torque_safe = peak_torque_joint <= env.torque_limits.unsqueeze(0) + 1e-5
    stable = sustained & (final_qd_rms < 1.0) & (final_base_ang < 0.5)
    safe = stable & velocity_safe & torque_safe.all(1) & ~rom_violation
    records.append(
      {
        "class": reset_class,
        "recovery": stood & ~fell,
        "sustained": sustained,
        "stable": stable,
        "safe": safe,
        "time": time_to_stand,
        "peak_qd": peak_qd_joint,
        "overspeed_fraction": overspeed_count / (sample_count[:, None] * cfg.control.decimation).clamp_min(1),
        "peak_torque": peak_torque_joint,
        "saturation_fraction": saturation_count / sample_count[:, None].clamp_min(1),
        "rom_violation": rom_violation,
        "pose_rms": final_pose_rms,
        "qd_rms": final_qd_rms,
        "base_ang": final_base_ang,
        "tilt_deg": final_tilt,
      }
    )
    env.reset_idx(torch.arange(num_envs, device=device))
    obs, *_ = env.step(torch.zeros(num_envs, env.num_actions, device=device))

  cat = {key: torch.cat([r[key] for r in records]) for key in records[0]}
  per_orientation = {}
  for idx, name in enumerate(RESET_CLASSES):
    mask = cat["class"] == idx
    per_orientation[name] = {
      "episodes": int(mask.sum()),
      "recovery": float(cat["recovery"][mask].float().mean()),
      "sustained": float(cat["sustained"][mask].float().mean()),
      "stable": float(cat["stable"][mask].float().mean()),
      "safe": float(cat["safe"][mask].float().mean()),
    }
  per_joint_qd = cat["peak_qd"].max(0).values
  per_joint_torque = cat["peak_torque"].max(0).values
  per_joint_sat = cat["saturation_fraction"].mean(0)
  return {
    "checkpoint": str(checkpoint.resolve()),
    "asset_sha256": asset_sha256(),
    "settings": {
      "num_envs": num_envs,
      "episodes": episodes,
      "seed": seed,
      "dr": dr,
      "pull_force": False,
      "action_scale": action_scale,
      "episode_s": episode_s,
      "velocity_sampling": "5 ms HoST control substep",
    },
    "overall": {
      "recovery": float(cat["recovery"].float().mean()),
      "sustained": float(cat["sustained"].float().mean()),
      "stable": float(cat["stable"].float().mean()),
      "safe": float(cat["safe"].float().mean()),
      "worst_orientation_recovery": min(v["recovery"] for v in per_orientation.values()),
      "velocity_violation_episode_fraction": float((cat["peak_qd"].max(1).values > 5).float().mean()),
      "overspeed_step_fraction": float(cat["overspeed_fraction"].mean()),
      "rom_violation_episode_fraction": float(cat["rom_violation"].float().mean()),
      "time_to_stand_s": _stats(cat["time"]),
      "peak_joint_velocity_rad_s": _stats(cat["peak_qd"].max(1).values),
      "final_posture_rms_rad": _stats(cat["pose_rms"]),
      "final_joint_velocity_rms_rad_s": _stats(cat["qd_rms"]),
      "final_base_angular_velocity_rad_s": _stats(cat["base_ang"]),
      "final_tilt_deg": _stats(cat["tilt_deg"]),
    },
    "per_orientation": per_orientation,
    "per_joint_peak_velocity_rad_s": dict(zip(env.dof_names, per_joint_qd.tolist(), strict=True)),
    "per_joint_peak_applied_torque_nm": dict(zip(env.dof_names, per_joint_torque.tolist(), strict=True)),
    "per_joint_torque_saturation_fraction": dict(zip(env.dof_names, per_joint_sat.tolist(), strict=True)),
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--episodes", type=int, default=5)
  p.add_argument("--dr", choices=("off", "on"), default="off")
  p.add_argument("--action-scale", type=float, default=0.25)
  p.add_argument("--episode-s", type=float, default=5.0)
  p.add_argument("--seed", type=int, default=20261011)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", type=Path)
  args = p.parse_args()
  result = rollout(
    args.checkpoint,
    args.num_envs,
    args.episodes,
    args.dr == "on",
    args.action_scale,
    args.episode_s,
    args.seed,
    args.device,
  )
  print(json.dumps(result, indent=2))
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
