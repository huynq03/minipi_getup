"""Lightweight checks of the FTSR task (no training).

Run with: WARP_CACHE_PATH=$PWD/.warp-cache-cu12 python -m minipi_getup.ftsr.validate

Checks: observation dimensions, Eq. 4 force/torque direction and finiteness, the
uprighting rotation vector, the stage-transition rule, the assist cutoff at t_tag,
the operational torque cap, and a no-NaN random rollout.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch


def _check(cond: bool, msg: str) -> None:
  print(("PASS " if cond else "FAIL ") + msg)
  if not cond:
    raise SystemExit(1)


def main(num_envs: int = 64, device: str = "cuda:0") -> None:
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from minipi_getup.ftsr.config.env_cfg import (
    ASSIST_END_ITERATION,
    ASSIST_F_MAX,
    SETTLE_STEPS,
    STEPS_PER_ITERATION,
  )
  from minipi_getup.ftsr.config.robot import OPERATIONAL_TORQUE_LIMIT
  from minipi_getup.ftsr.config.stage_rewards import STAGE_HEIGHTS, STAGE_WEIGHTS
  from minipi_getup.ftsr.mdp.assistance import WRENCH_KEY, uprighting_rotvec
  from minipi_getup.ftsr.mdp.stages import ftsr_stage_manager

  # 1. Uprighting rotation vector.
  upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  half = -math.pi / 4  # rotation of -pi/2 about y: supine
  supine = torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]])
  r_up, r_sup = uprighting_rotvec(upright), uprighting_rotvec(supine)
  _check(float(r_up.norm()) < 1e-6, "rotvec is zero when upright")
  _check(abs(float(r_sup.norm()) - math.pi / 2) < 1e-5, "rotvec norm = pi/2 lying")
  # Supine: body z points to world -x; rotating about +y by +pi/2 brings it to +z.
  _check(float(r_sup[0, 1]) > 1.5, "rotvec axis for supine is +y")

  # 2. Env construction and observation dimensions.
  cfg = load_env_cfg("Mjlab-FTSR-MiniPi")
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  obs, _ = env.reset()
  dims = {k: v.shape[-1] for k, v in obs.items()}
  print("obs dims", dims)
  _check(
    dims == {"actor": 47, "student": 235, "teacher": 36, "critic": 51},
    "observation dims (actor 47, student 5x47, teacher 36, critic 51)",
  )

  # 3. Settle window: no force; afterwards a finite upward force on lying robots.
  zero = torch.zeros(num_envs, env.action_manager.total_action_dim, device=device)
  env.step(zero)
  w = env.extras[WRENCH_KEY]
  _check(bool((w == 0).all()), "no assistance during the settle window")
  for _ in range(SETTLE_STEPS + 2):
    env.step(zero)
  w = env.extras[WRENCH_KEY]
  robot = env.scene["robot"]
  h = robot.data.body_link_pos_w[:, 0, 2]
  low = h < STAGE_HEIGHTS[0] - 0.02
  _check(bool(torch.isfinite(w).all()), "assist wrench finite")
  _check(bool((w[:, 2] >= 0).all()), "assist force points up (never down)")
  _check(bool((w[low, 2] > 1.0).all()), "lying robots get an upward force")
  _check(float(w[:, 2].max()) <= ASSIST_F_MAX + 1e-4, "force <= F_max")
  print(f"  force on lying robots: {w[low, 2].mean():.1f} N (F_max {ASSIST_F_MAX:.1f})")
  # The torque rotates the body z axis toward world z.
  quat = robot.data.body_link_quat_w[:, 0]
  ww, x, y, _ = quat.unbind(-1)
  zq = quat[:, 3]
  bz = torch.stack(
    (2 * (x * zq + ww * y), 2 * (y * zq - ww * x), 1 - 2 * (x * x + y * y)), -1
  )
  want = torch.stack((bz[:, 1], -bz[:, 0], torch.zeros_like(bz[:, 0])), -1)
  tilted = bz[:, 2] < 0.9
  dots = (w[tilted, 3:] * want[tilted]).sum(-1)
  _check(bool((dots >= -1e-6).all()), "assist torque rotates the torso upright")

  # 4. Random rollout: rewards, observations finite; torque within the envelope.
  monitor = env.action_manager.get_term("joint_pos").monitor
  monitor.clear()
  for _ in range(150):
    a = torch.randn_like(zero) * 2.0
    obs, rew, term, trunc, _ = env.step(a)
    assert torch.isfinite(rew).all(), "non-finite reward"
    for k, v in obs.items():
      assert torch.isfinite(v).all(), f"non-finite obs {k}"
  print("  random-action torque stats:", monitor.summary())
  _check(True, "150-step random rollout: rewards and observations finite")
  _check(
    monitor.summary()["torque_max"] <= OPERATIONAL_TORQUE_LIMIT * 1.05 + 0.05,
    "torque never exceeds the operational envelope (x1.05 motor-strength DR)",
  )

  # 5. Stage rule (2/3 of the population above h1 / h2), via a mocked height field.
  term = env.event_manager.get_term_cfg("stage").func
  assert isinstance(term, ftsr_stage_manager)

  def fake_env(heights: torch.Tensor):
    pos = torch.zeros(num_envs, 1, 3, device=device)
    pos[:, 0, 2] = heights
    data = SimpleNamespace(body_link_pos_w=pos)
    scene = {"robot": SimpleNamespace(data=data)}
    return SimpleNamespace(
      scene=scene,
      reward_manager=env.reward_manager,
      extras={},
      common_step_counter=0,
    )

  params = dict(cfg.events["stage"].params)
  params["asset_cfg"] = SimpleNamespace(name="robot", body_ids=[0])

  def run(frac_h1: float, frac_h2: float) -> int:
    hts = torch.full((num_envs,), 0.1, device=device)
    n1, n2 = int(frac_h1 * num_envs), int(frac_h2 * num_envs)
    hts[:n1] = STAGE_HEIGHTS[0] + 0.01
    hts[:n2] = STAGE_HEIGHTS[1] + 0.01
    fe = fake_env(hts)
    # Start a fresh decision window (the rollout above left it part-filled).
    term._count, term._sum_s1, term._sum_s2 = 0, 0.0, 0.0
    for _ in range(STEPS_PER_ITERATION):
      term(fe, None, **params)
    return term.stage

  _check(run(0.5, 0.0) == 0, "stage r_u while |S1| <= 2/3 N")
  _check(run(0.75, 0.2) == 1, "stage r_s once |S1| > 2/3 N")
  w_s = env.reward_manager.get_term_cfg("base_height").weight
  _check(w_s == STAGE_WEIGHTS["base_height"][1], "r_s weights applied")
  _check(run(0.9, 0.8) == 2, "stage r_w once |S2| > 2/3 N")
  _check(
    env.reward_manager.get_term_cfg("track_lin_vel").weight
    == STAGE_WEIGHTS["track_lin_vel"][2],
    "r_w weights applied (velocity tracking on)",
  )
  _check(run(0.1, 0.0) == 0, "stage falls back to r_u if the population drops")

  # 6. Assist cutoff: from t_tag on, the wrench is exactly zero.
  env.common_step_counter = ASSIST_END_ITERATION * STEPS_PER_ITERATION
  env.step(zero)
  _check(bool((env.extras[WRENCH_KEY] == 0).all()), "assist exactly zero at t_tag")
  print("all checks passed")
  env.close()


if __name__ == "__main__":
  main()
