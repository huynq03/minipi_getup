"""Interactive joint-range viewer for the staged-recovery asset (viser, no physics).

One slider per joint, limited to the XML range; the robot is posed kinematically
with the base fixed in the air. Shows the true knee flexion angle (calf 0 = 37 deg
flexed, -0.65 = straight, below = bent backwards) and optionally the collision
shells.

  python -m minipi_getup.host_staged_recovery.view_joints --port 8090
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import viser

from minipi_getup.host_staged_recovery.asset import ASSET_XML


def knee_flexion_deg(m: mujoco.MjModel, d: mujoco.MjData, side: str) -> float:
  anchor = lambda n: d.xanchor[m.joint(n).id]  # noqa: E731
  thigh = anchor(f"{side}_calf_joint") - anchor(f"{side}_hip_pitch_joint")
  shank = anchor(f"{side}_ankle_pitch_joint") - anchor(f"{side}_calf_joint")
  return float(np.degrees(np.arctan2(np.cross(thigh, shank)[1], np.dot(thigh, shank))))


SOLE_BOTTOM_Z = -0.048  # sole capsule bottoms in the ankle_roll_link frame
FEET = ("l_ankle_roll_link", "r_ankle_roll_link")


def standing_foot_targets(m: mujoco.MjModel, d: mujoco.MjData):
  """Foot body poses with all joints at 0 and the soles resting on z = 0."""
  d.qpos[:] = m.qpos0
  d.qpos[2] = 1.0
  mujoco.mj_kinematics(m, d)
  d.qpos[2] = 1.0 - (d.xpos[m.body(FEET[0]).id][2] + SOLE_BOTTOM_Z)
  mujoco.mj_kinematics(m, d)
  return d.qpos[2], [(d.xpos[m.body(f).id].copy(), d.xquat[m.body(f).id].copy()) for f in FEET]


def solve_feet_planted(m, d, targets, free_joints, iters=60, damping=1e-4):
  """Damped least-squares IK: move base x/z and `free_joints` so both feet keep
  their target pose (flat, in place) while the torso stays upright and the other
  joints stay where the sliders put them. Returns the residual (mm, deg)."""
  dofs = [0, 2] + [m.jnt_dofadr[m.joint(j).id] for j in free_joints]
  qadr = [0, 2] + [m.jnt_qposadr[m.joint(j).id] for j in free_joints]
  rng = [None, None] + [m.jnt_range[m.joint(j).id] for j in free_joints]
  jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  q_cur, dq = np.zeros(4), np.zeros(3)
  for _ in range(iters):
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    rows, errs = [], []
    for f, (pos, quat) in zip(FEET, targets, strict=True):
      b = m.body(f).id
      mujoco.mj_jacBody(m, d, jacp, jacr, b)
      mujoco.mju_mat2Quat(q_cur, d.xmat[b])
      mujoco.mju_subQuat(dq, quat, q_cur)  # body-frame rotation error
      errs += [pos - d.xpos[b], d.xmat[b].reshape(3, 3) @ dq]
      rows += [jacp[:, dofs], jacr[:, dofs]]
    J, e = np.vstack(rows), np.concatenate(errs)
    step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(e)), e)
    for a, r, v in zip(qadr, rng, step, strict=True):
      d.qpos[a] = d.qpos[a] + v if r is None else np.clip(d.qpos[a] + v, r[0], r[1])
  mujoco.mj_kinematics(m, d)
  pos_err = max(np.linalg.norm(t[0] - d.xpos[m.body(f).id]) for f, t in zip(FEET, targets, strict=True))
  ang_err = 0.0
  for f, (_, quat) in zip(FEET, targets, strict=True):
    mujoco.mju_mat2Quat(q_cur, d.xmat[m.body(f).id])
    mujoco.mju_subQuat(dq, quat, q_cur)
    ang_err = max(ang_err, np.degrees(np.linalg.norm(dq)))
  return pos_err * 1000, ang_err


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--port", type=int, default=8090)
  args = p.parse_args()

  m = mujoco.MjModel.from_xml_path(str(ASSET_XML))
  d = mujoco.MjData(m)
  server = viser.ViserServer(port=args.port)
  server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)

  handles = []  # (geom id, handle, is_collision)
  for g in range(m.ngeom):
    is_mesh = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    is_box = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_contype[g] != 0
    if is_mesh:
      mid = m.geom_dataid[g]
      verts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      faces = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
      h = server.scene.add_mesh_simple(f"/robot/mesh_{g}", verts, faces, color=(190, 190, 190))
      handles.append((g, h, False))
    elif is_box:
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      color = (255, 120, 60) if "shell" in name else (80, 200, 255)
      h = server.scene.add_box(f"/robot/box_{g}", dimensions=2 * m.geom_size[g], color=color,
                               opacity=0.35, visible=False)
      handles.append((g, h, True))

  hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
  base_z0, foot_targets = standing_foot_targets(m, d)
  ik_joints = [f"{s}_{k}_joint" for s in ("l", "r") for k in ("hip_pitch", "ankle_pitch")]
  readout = server.gui.add_markdown("")
  planted = server.gui.add_checkbox("Feet planted (IK: base, hip pitch, ankle pitch follow)", True)
  show_coll = server.gui.add_checkbox("Show collision boxes (orange = shells)", False)
  link = server.gui.add_checkbox("Mirror left/right calf", True)
  sliders = {}
  for j, n in zip(hinge, names, strict=True):
    lo, hi = m.jnt_range[j]
    sliders[n] = server.gui.add_slider(n.replace("_joint", ""), float(lo), float(hi), 0.01, 0.0)
  calf_lo, calf_hi = (float(v) for v in m.jnt_range[m.joint("l_calf_joint").id])
  sweep = server.gui.add_button(f"Sweep calf {calf_lo:.2f} -> {calf_hi:.2f}")

  def update(_=None):
    d.qpos[:] = m.qpos0
    d.qpos[2] = base_z0 if planted.value else 0.45
    for j, n in zip(hinge, names, strict=True):
      d.qpos[m.jnt_qposadr[j]] = sliders[n].value
    ik_note = ""
    if planted.value:
      for n in ik_joints:  # IK owns these; start from the default pose each time
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = 0.0
      pos_mm, ang_deg = solve_feet_planted(m, d, foot_targets, ik_joints)
      for n in ik_joints:
        sliders[n].disabled = True
      solved = ", ".join(f"{n.replace('_joint', '')} {d.qpos[m.jnt_qposadr[m.joint(n).id]]:+.2f}" for n in ik_joints)
      ok = "feet held" if pos_mm < 2 and ang_deg < 1 else "**cannot hold feet: joint limits reached**"
      ik_note = (f"  \nbase height {d.qpos[2]:.3f} m (standing default {base_z0:.3f}), forward {d.qpos[0]*100:+.1f} cm"
                 f"  \nIK: {solved}  \nfoot error {pos_mm:.1f} mm / {ang_deg:.1f} deg -> {ok}")
    else:
      for n in ik_joints:
        sliders[n].disabled = False
    mujoco.mj_kinematics(m, d)
    quat = np.zeros(4)
    for g, h, coll in handles:
      mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
      h.position = d.geom_xpos[g].copy()
      h.wxyz = quat.copy()
      if coll:
        h.visible = show_coll.value
    rows = []
    for s in ("l", "r"):
      c = sliders[f"{s}_calf_joint"].value
      ang = knee_flexion_deg(m, d, s)
      state = "straight" if abs(c + 0.65) < 0.02 else ("**bent backwards**" if c < -0.65 else "flexed")
      rows.append(f"`{s}` calf {c:+.2f} rad -> knee {ang:+.0f} deg ({state})")
    readout.content = ("  \n".join(rows) + "  \ncalf 0 = 37 deg flexed (default), -0.65 = straight, -2.95 = 132 deg backwards"
                       + ik_note)

  def on_calf(side):
    def cb(_):
      if link.value:
        other = sliders[f"{'r' if side == 'l' else 'l'}_calf_joint"]
        if abs(other.value - sliders[f"{side}_calf_joint"].value) > 1e-6:
          other.value = sliders[f"{side}_calf_joint"].value
      update()
    return cb

  for n, s in sliders.items():
    s.on_update(on_calf(n[0]) if n.endswith("calf_joint") else update)
  show_coll.on_update(update)
  planted.on_update(update)

  @sweep.on_click
  def _(_):
    for c in np.concatenate([np.linspace(0, calf_lo, 60), np.linspace(calf_lo, calf_hi, 120), np.linspace(calf_hi, 0, 60)]):
      sliders["l_calf_joint"].value = float(c)
      sliders["r_calf_joint"].value = float(c)
      update()
      time.sleep(1 / 30)

  update()
  print(f"[view_joints] {ASSET_XML.name} on http://localhost:{args.port}", flush=True)
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()
