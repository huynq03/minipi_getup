"""Narrow flat-footed squat poses for squat-start resets (rev 12).

Both soles flat at the default standing positions (under the hips), the COM
horizontally between the feet, knees bent by `calf`, no self-collision. Solved
once on the CPU with damped Gauss-Newton on (base x, base z, base pitch, hip
pitch L/R, ankle pitch L/R).
"""

from __future__ import annotations

import mujoco
import numpy as np

from minipi_getup.host_staged_recovery.geometry import SOLE_CORNERS_LOCAL

FEET = ("l_ankle_roll_link", "r_ankle_roll_link")
SOLE_CENTRE = np.array(SOLE_CORNERS_LOCAL).mean(0)


def _sole_targets(m, d):
  d.qpos[:] = m.qpos0
  d.qpos[2] = 1.0
  mujoco.mj_kinematics(m, d)
  bodies = [m.body(f).id for f in FEET]
  low = min((d.xpos[b] + d.xmat[b].reshape(3, 3) @ SOLE_CENTRE)[2] for b in bodies)
  d.qpos[2] = 1.0 - low + 0.001  # corners are the capsule bottoms already
  mujoco.mj_kinematics(m, d)
  return float(d.qpos[2]), [(d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in bodies]


def _residual(m, d, targets, adr, x):
  d.qpos[0], d.qpos[2] = x[0], x[1]
  d.qpos[3:7] = (np.cos(x[2] / 2), 0.0, np.sin(x[2] / 2), 0.0)
  for a, v in zip(adr, x[3:], strict=True):
    d.qpos[a] = v
  mujoco.mj_kinematics(m, d)
  mujoco.mj_comPos(m, d)
  res = []
  for f, (pos, rot) in zip(FEET, targets, strict=True):
    b = m.body(f).id
    res.append(d.xpos[b] - pos)
    err = np.zeros(3)
    mujoco.mju_subQuat(err, _quat(rot), _quat(d.xmat[b].reshape(3, 3)))
    res.append(0.05 * err)  # ~5 cm per radian
  com_x = d.subtree_com[m.body("base_link").id][0]
  mid_x = np.mean([(d.xpos[m.body(f).id] + d.xmat[m.body(f).id].reshape(3, 3) @ SOLE_CENTRE)[0] for f in FEET])
  res.append([com_x - mid_x])
  return np.concatenate(res)


def _quat(rot):
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, rot.reshape(-1))
  return q


def build_squat_library(xml: str, dof_names: list[str], calves=None, iters: int = 80):
  """Returns dict of numpy arrays: base_z (P,), base_pitch (P,), base_x (P,),
  dof_pos (P, n_dof) in `dof_names` order, calf (P,)."""
  m = mujoco.MjModel.from_xml_path(xml)
  d = mujoco.MjData(m)
  z0, targets = _sole_targets(m, d)
  free = ["l_hip_pitch_joint", "r_hip_pitch_joint", "l_ankle_pitch_joint", "r_ankle_pitch_joint"]
  adr = [m.jnt_qposadr[m.joint(n).id] for n in free]
  lo = np.array([-np.inf, 0.05, -1.2] + [m.jnt_range[m.joint(n).id][0] for n in free])
  hi = np.array([np.inf, 0.6, 1.2] + [m.jnt_range[m.joint(n).id][1] for n in free])
  out = {k: [] for k in ("base_z", "base_pitch", "base_x", "dof_pos", "calf")}
  for calf in np.linspace(0.3, 1.65, 28) if calves is None else calves:
    d.qpos[:] = m.qpos0
    for s in "lr":
      d.qpos[m.jnt_qposadr[m.joint(f"{s}_calf_joint").id]] = calf
    x = np.array([0.0, z0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
        or abs(r[12]) > 0.01:
      continue
    mujoco.mj_forward(m, d)
    if d.ncon:  # self-collision (the asset has no floor)
      continue
    q = np.array([d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in dof_names])
    out["base_z"].append(x[1])
    out["base_pitch"].append(x[2])
    out["base_x"].append(x[0])
    out["dof_pos"].append(q)
    out["calf"].append(calf)
  return {k: np.array(v) for k, v in out.items()}
