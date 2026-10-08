"""Derive support references by bounded IK, never by guessed joint angles.

The solutions are an unordered set of feasible poses for soft shaping, not a
commanded trajectory. Dynamic supine recovery feasibility is NOT established by IK.
"""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

XML = (
  Path(__file__).resolve().parents[1]
  / "asset_zoo/robots/hightorque_minipi/xmls/cl_pai.xml"
)


def bottom(model, data, geom):
  kind = model.geom_type[geom]
  if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return (
      data.geom_xpos[geom, 2]
      - model.geom_size[geom, 0]
      - model.geom_size[geom, 1] * abs(data.geom_xmat[geom].reshape(3, 3)[2, 2])
    )
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    return data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
  raise ValueError("Unexpected collision primitive")


@lru_cache(maxsize=1)
def calibrate():
  m = mujoco.MjModel.from_xml_path(str(XML))
  d = mujoco.MjData(m)
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
  feet = [
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s + "_ankle_roll_link")
    for s in ("l", "r")
  ]
  sites = [
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s + "_foot") for s in ("l", "r")
  ]
  collision = np.flatnonzero(m.geom_contype)
  foot_geoms = [g for g in collision if m.geom_bodyid[g] in feet]
  # Nominal zero joints are specified by the existing asset-zoo HOME_KEYFRAME.
  d.qpos[:] = 0
  d.qpos[3] = 1
  mujoco.mj_forward(m, d)
  standing_height = -min(bottom(m, d, g) for g in foot_geoms)
  d.qpos[2] = standing_height
  mujoco.mj_forward(m, d)
  foot_xy = d.site_xpos[sites, :2].copy()
  nominal_com = d.subtree_com[1].copy()
  # Stable reference stance must put COM within the two-foot footprint.
  assert foot_xy[:, 0].min() - 0.065 < nominal_com[0] < foot_xy[:, 0].max() + 0.065
  assert foot_xy[:, 1].min() - 0.02 < nominal_com[1] < foot_xy[:, 1].max() + 0.02
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  poses, heights, residuals = [], [], []
  q0 = np.zeros(12)
  for ratio in (1.0, 0.9, 0.8, 0.7, 0.65):
    height = ratio * standing_height

    def residual(q, height=height):
      d.qpos[2] = height
      d.qpos[7:] = q
      mujoco.mj_forward(m, d)
      pos = (d.site_xpos[sites] - np.c_[foot_xy, np.zeros(2)]).ravel()
      normals = np.concatenate(
        [d.xmat[b].reshape(3, 3)[:, 2] - [0, 0, 1] for b in feet]
      )
      # Tiny reference regularizer selects the nearby branch, not a motion script.
      return np.r_[pos * 10, normals, q * 0.002]

    sol = least_squares(
      residual, q0, bounds=(lo + 0.025, hi - 0.025), xtol=1e-11, ftol=1e-11, gtol=1e-11
    )
    residual(sol.x)
    err = np.max(np.abs(d.site_xpos[sites] - np.c_[foot_xy, np.zeros(2)]))
    clearance = min(bottom(m, d, g) for g in collision if g not in foot_geoms)
    penetration = max([max(0.0, -x.dist) for x in d.contact] or [0.0])
    if err < 0.004 and clearance > 0.003 and penetration < 0.002:
      poses.append(sol.x.copy())
      heights.append(height)
      residuals.append(float(err))
      q0 = sol.x.copy()
  assert len(poses) >= 3, "No sufficiently broad collision-free support set"
  return dict(
    joint_names=names,
    limits=m.jnt_range[1:].tolist(),
    standing_height=float(standing_height),
    support_heights=heights,
    support_poses=np.array(poses).tolist(),
    ik_error_m=residuals,
    nominal_pose=np.zeros(12).tolist(),
    foot_xy=foot_xy.tolist(),
    mass=float(m.body_mass.sum()),
  )


if __name__ == "__main__":
  import json

  print(json.dumps(calibrate(), indent=2))
