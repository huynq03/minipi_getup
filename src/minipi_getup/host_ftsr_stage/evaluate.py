"""Deterministic supine evaluation of stage checkpoints; no parameter updates."""

import argparse
import hashlib
from pathlib import Path

import torch

from .geometry import XML


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", type=Path, required=True)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--episodes", type=int, default=3)
  ap.add_argument("--action-scale", type=float)
  ap.add_argument("--dr", choices=("off", "on"), default="off")
  a = ap.parse_args()
  if min(a.num_envs, a.episodes) < 1:
    ap.error("Counts must be positive")
  s = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
  info = s.get("infos") or {}
  if info.get("experiment") != "host_ftsr_stage_v1":
    ap.error("Use a host_ftsr_stage_v1 checkpoint")
  if info.get("asset_sha256") != hashlib.sha256(XML.read_bytes()).hexdigest():
    ap.error("Asset changed since training")
  from warp._src.build import init_kernel_cache

  init_kernel_cache("/tmp/host-supine-narrow-warp")
  from minipi_getup.host.evaluate import load_policy

  from .config import StageCfg
  from .env import StageEnv

  torch.set_num_threads(4)
  torch.manual_seed(0)
  c = StageCfg()
  c.noise.add_noise = False
  c.curriculum.pull_force = False
  c.curriculum.force = 0
  c.control.action_scale = (
    a.action_scale
    if a.action_scale is not None
    else float(info["curriculum_state"]["action_rescale"].mean())
  )
  if a.dr == "off":
    for k in dir(c.domain_rand):
      if k.startswith("randomize_") or k == "delay":
        setattr(c.domain_rand, k, False)
  e = StageEnv(c, a.num_envs)
  policy = load_policy(str(a.checkpoint), e)
  from dataclasses import asdict

  from .config import StageSettings

  if info.get("stage_settings") != asdict(StageSettings()):
    ap.error("Stage settings differ from checkpoint")
  stats = []
  with torch.no_grad():
    for _ep in range(a.episodes):
      obs, _ = e.reset()
      alive = torch.ones(a.num_envs, device=e.device, dtype=torch.bool)
      any_up = torch.zeros_like(alive)
      last = {}
      for _ in range(int(e.max_episode_length) + 2):
        obs, _, _, done, _ = e.step(policy.act_inference(obs))
        # step auto-resets done rows: freeze their pre-reset snapshot.
        alive &= ~done.bool()
        if not alive.any():
          break
        logic = e.stage_logic
        up = (
          (logic.last["up"] > 0.8)
          & (e.root_states[:, 2] > 0.3173)
          & (e.real_episode_length_buf > e.unactuated_time)
        )
        any_up |= up & alive
        current = dict(
          any_upright=any_up.float(),
          upright_any_method=(logic.upright_hold >= logic.cfg.hold_s).float(),
          stable=(logic.hold >= logic.cfg.hold_s).float(),
          direct=(
            (logic.hold >= logic.cfg.hold_s) & ~logic.prone_seen & ~logic.side_seen
          ).float(),
          physical=(
            (logic.hold >= logic.cfg.hold_s)
            & (e.max_overshoot <= logic.cfg.joint_overshoot_tolerance)
          ).float(),
          prone=logic.prone_seen.float(),
          max_side_deg=logic.max_side * 180 / torch.pi,
          roll_events=logic.roll_events,
          overshoot=e.max_overshoot,
          peak_torque=e.peak_torque.max(-1).values,
          peak_qdot=e.peak_joint_vel.max(-1).values,
          saturation=e.sat_count.sum(-1) / (12 * e.sample_count.clamp(min=1)),
          transitions=logic.transitions,
          regressions=logic.regressions,
        )
        for i in range(3):
          current[f"stage{i + 1}_seconds"] = logic.time_in_stage[:, i]
        for i, n in enumerate(e.dof_names):
          current["min_" + n] = e.q_min[:, i]
          current["max_" + n] = e.q_max[:, i]
        for k, v in current.items():
          if k not in last:
            last[k] = v.clone()
          else:
            last[k] = torch.where(alive, v, last[k])
      stats.append(last)
  print(
    f"checkpoint={a.checkpoint}; force OFF; DR={a.dr}; beta={c.control.action_scale:.5f}; samples={a.num_envs * a.episodes}"
  )
  print(
    "Deterministic action mean; original HoST beta-observation jitter is retained. DR OFF gives repeated nominal drops, not broad robustness."
  )
  for k in stats[0]:
    values = torch.cat([d[k] for d in stats])
    print(
      f"{k}: mean={values.mean().item():.6f}, min={values.min().item():.6f}, max={values.max().item():.6f}"
    )


if __name__ == "__main__":
  main()
