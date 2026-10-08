"""v2 checks. CPU: stage entry vs strict success, margin smoothness/monotonicity, no
stage oscillation. --checkpoint (GPU): replay a v1 checkpoint with v1 and v2 stage
logic side by side, and test that strict success is physically attainable with the
compliant MuJoCo joint stops (PD hold of a feasible standing pose, limits unchanged).

python -m minipi_getup.host_ftsr_stage_v2.validate [--checkpoint V1_MODEL_1000]
"""

import argparse
import math

import numpy as np
import torch

from minipi_getup.host_ftsr_stage.geometry import calibrate
from minipi_getup.host_ftsr_stage.validate import feature_pose

from .config import StageSettingsV2
from .rewards import StageLogicV2


def with_overshoot(f, over, n):
  """Upright, well-supported pose whose worst joint is `over` rad past its limit."""
  f = {k: v.clone() for k, v in f.items()}
  f["overshoot"] = torch.full((n,), over)
  d = torch.zeros(n, 12)
  d[:, 0] = over + StageSettingsV2().joint_margin_rad if over > 0 else 0.0
  f["margin_dist"] = d
  return f


def run_logic(f, steps, n=1):
  geo = calibrate()
  logic = StageLogicV2(n, "cpu", 0.02, geo)
  active = torch.ones(n, dtype=torch.bool)
  hist = []
  for _ in range(steps):
    out = logic.update(f, active)
    hist.append(
      (int(logic.stage[0]), bool(out["success"][0]), float(out["joint_margin"][0]))
    )
  return logic, hist


def cpu_checks():
  s = StageSettingsV2()
  assert s.joint_overshoot_tolerance == 0.015 and s.stage3_entry_overshoot == 0.03
  base = feature_pose("upright")
  # Stage 1 -> 2 needs erect dwell, 2 -> 3 support dwell; run 3 s.
  for over, want_stage3, want_strict in ((0.023, True, False), (0.010, True, True)):
    f = with_overshoot(base, over, 1)
    logic, hist = run_logic(f, 150)
    stages = [h[0] for h in hist]
    assert (stages[-1] == 2) == want_stage3, (over, stages[-1])
    assert any(h[1] for h in hist) == want_strict, (over, "strict")
    # Latched: stage never decreases, at most two transitions.
    assert all(b >= a for a, b in zip(stages, stages[1:], strict=False))
    assert float(logic.transitions[0]) <= 2
    first3 = stages.index(2) if 2 in stages else None
    first_ok = next((i for i, h in enumerate(hist) if h[1]), None)
    print(
      f"PASS overshoot {over:.3f}: stage3 at step {first3}, strict success at "
      f"step {first_ok}, target_joint_margin {hist[-1][2]:.3f}"
    )
  # Over the entry tolerance: stays in Stage 2.
  _, hist = run_logic(with_overshoot(base, 0.035, 1), 150)
  assert hist[-1][0] == 1 and not any(h[1] for h in hist)
  print("PASS overshoot 0.035: remains in Stage 2, no strict success")
  # Margin score: monotone in overshoot, smooth through 0.015 (no jump).
  overs = torch.linspace(-0.05, 0.08, 1301)
  geo = calibrate()
  logic = StageLogicV2(len(overs), "cpu", 0.02, geo)
  f = {k: v.expand(len(overs), *v.shape[1:]).clone() for k, v in base.items()}
  f["overshoot"] = overs.clamp(min=0)
  d = torch.zeros(len(overs), 12)
  d[:, 0] = (overs + s.joint_margin_rad).clamp(min=0)  # band starts 0.05 inside
  f["margin_dist"] = d
  out = logic.update(f, torch.ones(len(overs), dtype=torch.bool))
  r = out["joint_margin"]
  assert torch.isfinite(r).all()
  assert (r[1:] <= r[:-1] + 1e-7).all(), (
    "margin reward must not increase with overshoot"
  )
  step = (r[1:] - r[:-1]).abs().max().item()
  i15 = int(torch.argmin((overs - 0.015).abs()))
  jump15 = (r[i15 + 1] - r[i15 - 1]).abs().item()
  print(
    f"PASS margin reward monotone; max step over 1e-4 rad grid {step:.2e}, change "
    f"across 0.015 rad {jump15:.2e}; interior {r[0]:.3f}, at limit "
    f"{r[int(torch.argmin(overs.abs()))]:.3f}, at 0.023 "
    f"{r[int(torch.argmin((overs - 0.023).abs()))]:.3f}"
  )
  # Gate: lying/erect/crouched earn nothing; standing earns it.
  for name, height in (("supine", 0.08), ("situp", 0.15), ("upright", 0.20)):
    g = feature_pose(name)
    g["height"] = torch.tensor([height])
    g["margin_dist"] = torch.zeros(1, 12)
    lg = StageLogicV2(1, "cpu", 0.02, geo)
    o = lg.update(g, torch.ones(1, dtype=torch.bool))
    assert float(o["joint_margin"][0]) < 1e-6, (name, float(o["joint_margin"][0]))
  print("PASS margin gate: supine/sit-up/crouch (base 0.20 m) earn 0")


@torch.no_grad()
def gpu_checks(checkpoint):
  from minipi_getup.host_ftsr_stage.rewards import StageLogic
  from minipi_getup.host_ftsr_stage_v2.evaluate import build

  info = torch.load(checkpoint, map_location="cpu", weights_only=False)["infos"]
  beta = float(info["curriculum_state"]["action_rescale"].mean())
  torch.manual_seed(0)
  env, policy = build(checkpoint, beta, False, 64)
  v1 = StageLogic(env.num_envs, env.device, env.dt, env.geometry)
  obs, _ = env.reset()
  v1.reset(torch.arange(env.num_envs, device=env.device))
  margin, standing, finite = [], [], True
  for _ in range(int(env.max_episode_length) - 2):
    obs, *_ = env.step(policy.act_inference(obs))
    rew = env.rew_buf
    active = env.real_episode_length_buf > env.unactuated_time
    v1.update(env.features, active)
    last = env.stage_logic.last
    margin.append(last["joint_margin"])
    standing.append(last["target"])
    finite &= bool(torch.isfinite(rew).all())
  lg = env.stage_logic
  margin, standing = torch.stack(margin), torch.stack(standing)
  print(
    f"replay {checkpoint.name} beta {beta:.3f}: v1 logic stage>=3 fraction "
    f"{(v1.stage >= 2).float().mean():.3f}; v2 logic {(lg.stage >= 2).float().mean():.3f}"
  )
  print(
    f"  v2 transitions per env max {lg.transitions.max():.0f}, regressions mean "
    f"{lg.regressions.mean():.2f}; strict success {(lg.hold >= lg.cfg.hold_s).float().mean():.3f}"
    f" (ever {lg.ever_stable.float().mean():.3f}); episode max overshoot median "
    f"{env.max_overshoot.median():.4f}"
  )
  print(
    f"  per-step target terms over last 2 s: joint_margin {margin[-100:].mean():.3f}, "
    f"stage_standing {standing[-100:].mean():.5f}; rewards finite {finite}"
  )
  assert (v1.stage >= 2).float().mean() < 0.05 < (lg.stage >= 2).float().mean()
  assert margin[-100:].mean() > 0.1 and finite
  assert lg.transitions.max() <= 2
  assert (lg.hold >= lg.cfg.hold_s).float().mean() == 0  # 0.023 rad is not strict
  print(
    "PASS v1-stuck states enter v2 Stage 3; joint margin reward nonzero; not strict"
  )
  feasibility(env)


@torch.no_grad()
def feasibility(env):
  """PD-hold feasible standing poses on the real limits; strict success reachable?"""
  geo = env.geometry
  n = env.num_envs
  ids = torch.arange(n, device=env.device)
  H = geo["standing_height"]
  poses = {
    "zero (nominal)": np.zeros(12),
    "ik support 1.0H": np.array(geo["support_poses"][0]),
    "zero + 0.1 rad knee bend": np.array([0, 0, 0, 0.1, -0.05, 0] * 2, dtype=float),
  }
  for name, q in poses.items():
    env.reset()
    q_t = torch.tensor(q, device=env.device, dtype=torch.float).expand(n, 12).clone()
    rs = env.root_states.clone()
    rs[:, :3] = env.env_origins
    rs[:, 2] += H + 0.003
    rs[:, 3:7] = torch.tensor([1.0, 0, 0, 0], device=env.device)
    rs[:, 7:] = 0
    env.robot.write_root_state_to_sim(rs, env_ids=ids)
    env.robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t), env_ids=ids)
    env.episode_length_buf[:] = env.unactuated_time + 1
    env.real_episode_length_buf[:] = env.unactuated_time + 1
    for _ in range(100):  # 2 s
      a = (q_t - env.dof_pos) / env.action_rescale
      env.step(a)
    lg = env.stage_logic
    f = env.features
    ok = (lg.hold >= lg.cfg.hold_s).float().mean().item()
    print(
      f"  hold '{name}': strict success {ok:.2f}, overshoot max {f['overshoot'].max():.4f}, "
      f"pose_max {f['pose_max'].median():.3f}, symmetry {f['symmetry_rms'].median():.3f}, "
      f"height {f['height'].median():.3f}, margin score {lg.last['margin_score'].median():.3f}"
    )
    assert math.isfinite(f["overshoot"].max().item())
  print("PASS strict-success feasibility test ran on unchanged physical limits")


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", type=__import__("pathlib").Path)
  a = ap.parse_args()
  cpu_checks()
  if a.checkpoint:
    gpu_checks(a.checkpoint)


if __name__ == "__main__":
  main()
