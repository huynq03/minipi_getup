"""Displaced standing stances for stance-start resets (revision 13).

Upright torso, both soles flat on the ground but displaced from their default
place (wider and/or staggered fore-aft, like the stances the policy stands up
into), the COM horizontally between the feet, no self-collision. Solved once on
the CPU with damped Gauss-Newton on (base x, y, z, 12 leg joints).
"""

from __future__ import annotations

import mujoco
import numpy as np

from minipi_getup.host_staged_recovery.geometry import SOLE_CORNERS_LOCAL

FEET = ("l_ankle_roll_link", "r_ankle_roll_link")
SOLE_CENTRE = np.array(SOLE_CORNERS_LOCAL).mean(0)


def _quat(rot):
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, rot.reshape(-1))
  return q


def _default_feet(m, d):
  """Base height with the default pose's soles on the ground, and the default foot frames."""
  d.qpos[:] = m.qpos0
  d.qpos[2] = 1.0
  mujoco.mj_kinematics(m, d)
  bodies = [m.body(f).id for f in FEET]
  low = min((d.xpos[b] + d.xmat[b].reshape(3, 3) @ SOLE_CENTRE)[2] for b in bodies)
  d.qpos[2] = 1.0 - low + 0.001
  mujoco.mj_kinematics(m, d)
  return float(d.qpos[2]), [(d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in bodies]


def _residual(m, d, targets, adr, x):
  d.qpos[0], d.qpos[1], d.qpos[2] = x[0], x[1], x[2]
  d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for a, v in zip(adr, x[3:], strict=True):
    d.qpos[a] = v
  mujoco.mj_kinematics(m, d)
  mujoco.mj_comPos(m, d)
  res = []
  mid = np.zeros(2)
  for f, (pos, rot) in zip(FEET, targets, strict=True):
    b = m.body(f).id
    res.append(d.xpos[b] - pos)
    err = np.zeros(3)
    mujoco.mju_subQuat(err, _quat(rot), _quat(d.xmat[b].reshape(3, 3)))
    res.append(0.05 * err)  # ~5 cm per radian
    mid += 0.5 * (d.xpos[b] + d.xmat[b].reshape(3, 3) @ SOLE_CENTRE)[:2]
  res.append(d.subtree_com[m.body("base_link").id][:2] - mid)
  res.append(1e-3 * np.asarray(x[3:]))  # weak pull toward zero joints (redundancy)
  return np.concatenate(res)


def build_stance_library(xml: str, dof_names: list[str], n: int = 64, seed: int = 0,
                         extra_out=(0.02, 0.08), stagger=(-0.08, 0.08), iters: int = 60):
  """Returns dict of numpy arrays: base_xyz (P, 3) with the feet midpoint at the
  origin, dof_pos (P, n_dof) in `dof_names` order, offset (P, 2, 2) = per-foot
  (outward, forward) displacement from the default place."""
  m = mujoco.MjModel.from_xml_path(xml)
  d = mujoco.MjData(m)
  z0, default = _default_feet(m, d)
  joints = [f"{s}_{j}_joint" for s in "lr" for j in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")]
  adr = [m.jnt_qposadr[m.joint(j).id] for j in joints]
  lo = np.array([-np.inf, -np.inf, 0.15] + [m.jnt_range[m.joint(j).id][0] + 0.02 for j in joints])
  hi = np.array([np.inf, np.inf, 0.5] + [m.jnt_range[m.joint(j).id][1] - 0.02 for j in joints])
  rng = np.random.default_rng(seed)
  out = {"base_xyz": [], "dof_pos": [], "offset": []}
  tries = 0
  while len(out["base_xyz"]) < n and tries < 20 * n:
    tries += 1
    outward = rng.uniform(*extra_out, size=2)
    fwd = rng.uniform(*stagger)
    forward = np.array([fwd, -fwd]) * rng.uniform(0.3, 1.0) if rng.random() < 0.7 else rng.uniform(*stagger, size=2) * 0.3
    targets = []
    for i, (pos, rot) in enumerate(default):
      side = 1.0 if i == 0 else -1.0  # left foot outward = +y
      targets.append((pos + np.array([forward[i], side * outward[i], 0.0]), rot))
    x = np.concatenate([[0.0, 0.0, z0], np.zeros(len(adr))])
    for _ in range(iters):
      r = _residual(m, d, targets, adr, x)
      jac = np.zeros((len(r), len(x)))
      for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1e-5
        jac[:, k] = (_residual(m, d, targets, adr, x + dx) - r) / 1e-5
      step = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(len(x)), -jac.T @ r)
      x = np.clip(x + step, lo, hi)
    r = _residual(m, d, targets, adr, x)
    if np.abs(r[[0, 1, 2, 6, 7, 8]]).max() > 0.003 or np.abs(r[[3, 4, 5, 9, 10, 11]]).max() > 0.05 * 0.03 \
        or np.abs(r[12:14]).max() > 0.01:
      continue
    mujoco.mj_forward(m, d)
    if d.ncon:  # self-collision (the asset has no floor)
      continue
    mid = np.mean([t[0][:2] for t in targets], axis=0)
    out["base_xyz"].append([x[0] - mid[0], x[1] - mid[1], x[2]])
    out["dof_pos"].append([d.qpos[m.jnt_qposadr[m.joint(nm).id]] for nm in dof_names])
    out["offset"].append(np.stack([outward, forward], -1))
  return {k: np.array(v) for k, v in out.items()}
