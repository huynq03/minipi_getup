"""Read-only geometric audit of Mini-Pi v2; no training or XML modifications."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'src/minipi_getup/asset_zoo/robots/hightorque_minipi/xmls/cl_pai_v2.xml'
OUT = ROOT / 'reports/host/collision_v2' / hashlib.sha256(XML.read_bytes()).hexdigest()[:12]
OUT.mkdir(parents=True, exist_ok=True)
spec = mujoco.MjSpec.from_file(str(XML))
spec.worldbody.add_geom(name='audit_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[2, 2, .1], contype=1, conaffinity=1)
m = spec.compile()
d = mujoco.MjData(m)
js = np.array([j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE])
adr = m.jnt_qposadr[js]
lo, hi = m.jnt_range[js, 0] + .001, m.jnt_range[js, 1] - .001
names = [m.joint(int(j)).name for j in js]
feet = [m.body(s + '_ankle_roll_link').id for s in 'rl']
centre = np.array([-.0514, 0, -.048])
meshes = {int(m.geom_bodyid[g]): g for g in range(m.ngeom)
          if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}


def pose(q, z, x=0.):
  d.qpos[:] = m.qpos0
  d.qpos[0], d.qpos[2] = x, z
  d.qpos[adr] = q
  mujoco.mj_forward(m, d)


def mesh_points(g):
  mid = m.geom_dataid[g]
  v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
  return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def contacts():
  result = {}
  for c in d.contact:
    if c.dist >= -.0001:
      continue
    a, b = map(int, c.geom)
    key = (a, b)
    if key in result and result[key]['depth_mm'] >= -c.dist * 1000:
      continue
    row = {'geoms': [m.geom(a).name, m.geom(b).name], 'depth_mm': float(-c.dist * 1000),
           'floor': a == m.geom('audit_floor').id or b == m.geom('audit_floor').id}
    ba, bb = map(int, m.geom_bodyid[[a, b]])
    if ba in meshes and bb in meshes:
      gap = mujoco.mj_geomDistance(m, d, meshes[ba], meshes[bb], 1.0, None)
      row['visual_convex_hull_gap_mm'] = float(gap * 1000)
      row['proven_mesh_separated'] = bool(gap > .0001)
    result[key] = row
  return list(result.values())


def snapshot(q, z, x=0.):
  pose(q, z, x)
  points = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ centre for b in feet]
  return {'q': dict(zip(names, map(float, q), strict=True)), 'base_xyz': [float(x), 0., float(z)],
          'sole_centres': np.array(points).tolist(),
          'whole_body_com_world': d.subtree_com[m.body('base_link').id].tolist(),
          'com_minus_sole_midpoint_xy': (d.subtree_com[m.body('base_link').id][:2] - np.mean(points, axis=0)[:2]).tolist(),
          'foot_tilt_deg': [float(np.degrees(np.arccos(np.clip(d.xmat[b].reshape(3, 3)[2, 2], -1, 1)))) for b in feet],
          'torso_mesh_floor_gap_mm': float(mesh_points(meshes[m.body('base_link').id])[:, 2].min() * 1000),
          'contacts': contacts()}


def residual(q, z, x, y):
  pose(q, z)
  r = []
  for i, b in enumerate(feet):
    rot = d.xmat[b].reshape(3, 3)
    pt = d.xpos[b] + rot @ centre
    r.extend((pt - [x, (-1 if i == 0 else 1) * y, 0]) * 10)
    r.extend(rot[:, 2] - [0, 0, 1])
  r.extend(q * .0001)
  return np.array(r)


# Exact geometry fit: transform compiled mesh vertices into each shell's frame.
pose(np.zeros(12), .5)
fits = []
for g in range(m.ngeom):
  if 'shell_collision' not in m.geom(g).name:
    continue
  b = int(m.geom_bodyid[g])
  local = (mesh_points(meshes[b]) - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)
  excess = np.maximum(np.abs(local) - m.geom_size[g], 0.)
  fits.append({'shell': m.geom(g).name, 'body': m.body(b).name,
               'mesh_min_in_shell_mm': (local.min(0) * 1000).tolist(),
               'mesh_max_in_shell_mm': (local.max(0) * 1000).tolist(),
               'shell_half_size_mm': (m.geom_size[g] * 1000).tolist(),
               'max_vertex_outside_box_mm': float(np.linalg.norm(excess, axis=1).max() * 1000)})
zero_air = snapshot(np.zeros(12), .5)
pose(np.zeros(12), 1.)
zero_z = 1 - min((d.xpos[b] + d.xmat[b].reshape(3, 3) @ centre)[2] for b in feet)
zero_ground = snapshot(np.zeros(12), zero_z)

scans = []
for z in (.058, .08, .10, .12, .15):
  for x in (-.18, -.10, 0., .10, .18):
    for y in (.08, .12):
      for sign in (-1, 1):
        initial = np.zeros(12)
        initial[[3, 9]] = -1.8 if sign < 0 else 1.1
        initial[[0, 6]] = .7 * sign
        sol = least_squares(residual, np.clip(initial, lo, hi), bounds=(lo, hi),
                            args=(z, x, y), max_nfev=180, ftol=1e-8)
        row = snapshot(sol.x, z)
        row['target'] = {'height': z, 'foot_x': x, 'half_width': y}
        row['ik_residual'] = float(np.linalg.norm(residual(sol.x, z, x, y)[:12]))
        row['flat_ground_valid'] = (max(row['foot_tilt_deg']) < 1
                                   and max(abs(p[2]) for p in row['sole_centres']) < .001)
        scans.append(row)
  rows = [r for r in scans if r['target']['height'] == z]
  print('height', z, 'valid', sum(r['flat_ground_valid'] for r in rows),
        'valid without penetration', sum(r['flat_ground_valid'] and not r['contacts'] for r in rows), flush=True)

valid = [r for r in scans if r['flat_ground_valid'] and not r['contacts']]
sitting = min(valid, key=lambda r: (r['base_xyz'][2], r['ik_residual'])) if valid else None
blocked = min([r for r in scans if r['flat_ground_valid'] and any('shell' in g for c in r['contacts'] for g in c['geoms'])],
              key=lambda r: r['base_xyz'][2], default=None)
transitions = []
planted_transitions = []
if sitting:
  qsit = np.array([sitting['q'][n] for n in names])
  for t in np.linspace(0, 1, 101):
    row = snapshot(qsit * (1-t), sitting['base_xyz'][2] * (1-t) + zero_z * t)
    row['t'] = float(t)
    transitions.append(row)
  targets = np.array(sitting['sole_centres'])
  last = np.r_[qsit, 0.]

  def planted_residual(x, z):
    pose(x[:-1], z, x[-1])
    r = []
    for i, b in enumerate(feet):
      rot = d.xmat[b].reshape(3, 3)
      r.extend((d.xpos[b] + rot @ centre - targets[i]) * 10)
      r.extend(rot[:, 2] - [0, 0, 1])
    r.extend((x - last) * .0001)
    return np.array(r)

  for z in np.linspace(sitting['base_xyz'][2], zero_z, 61):
    sol = least_squares(planted_residual, last, bounds=(np.r_[lo, -.5], np.r_[hi, .5]),
                        args=(z,), max_nfev=120)
    error = float(np.linalg.norm(planted_residual(sol.x, z)[:12]))
    last = sol.x
    row = snapshot(last[:-1], z, last[-1])
    row['ik_residual'] = error
    planted_transitions.append(row)

# Reproducible ROM samples: identify enabled contacts and demonstrably over-wide proxies.
rng = np.random.default_rng(0)
pair_counts = Counter()
false_contacts = []
penetrating = 0
for i in range(2000):
  q = rng.uniform(lo, hi)
  row = snapshot(q, .5)
  penetrating += bool(row['contacts'])
  for c in row['contacts']:
    pair_counts[' / '.join(c['geoms'])] += 1
    if c.get('proven_mesh_separated') and any('shell' in g for g in c['geoms']) and len(false_contacts) < 10:
      false_contacts.append({'sample': i, 'q': row['q'], 'contact': c})

# Plot projected visual mesh and shell boxes, without changing the source asset.
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, row, title in zip(axes, [zero_ground, sitting, blocked],
                          ['Zero pose', 'Lowest collision-free flat-foot candidate', 'Blocked flat-foot candidate'], strict=True):
  if row is None:
    continue
  pose(np.array([row['q'][n] for n in names]), row['base_xyz'][2], row['base_xyz'][0])
  for g in meshes.values():
    v = mesh_points(g)[:, [0, 2]]
    hull = ConvexHull(v)
    ax.fill(v[hull.vertices, 0], v[hull.vertices, 1], color='gray', alpha=.25)
  for g in range(m.ngeom):
    if 'shell_collision' not in m.geom(g).name:
      continue
    v = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * m.geom_size[g]
    v = (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, [0, 2]]
    hull = ConvexHull(v)
    v = v[np.r_[hull.vertices, hull.vertices[0]]]
    ax.plot(v[:, 0], v[:, 1], color='darkorange', linewidth=.8)
  ax.axhline(0, color='black')
  ax.set_title(title + f"\nbase z={row['base_xyz'][2]:.3f} m")
  ax.set_aspect('equal')
  ax.set_xlabel('world X (m)')
  ax.set_ylabel('world Z (m)')
fig.tight_layout()
fig.savefig(OUT / 'poses_side.png', dpi=150)
plt.close(fig)
result = {'xml': str(XML), 'sha256': hashlib.sha256(XML.read_bytes()).hexdigest(),
          'method': 'Upright torso; 100 bounded IK attempts at five heights/two widths/five fore-aft offsets/two initial knee branches. Static contacts with a plane, depth threshold 0.1 mm. No dynamics, no proof of global feasibility. Visual mesh distance uses convex hulls: positive gap proves mesh separation; negative gap does not prove triangle-mesh penetration.',
          'shell_fit': fits, 'zero_air': zero_air, 'zero_ground': zero_ground,
          'sit_scan': scans, 'lowest_collision_free_candidate': sitting, 'blocked_candidate': blocked,
          'linear_sit_to_zero': transitions,
          'feet_planted_continuation': planted_transitions,
          'random_rom': {'seed': 0, 'samples': 2000, 'penetrating_samples': penetrating,
                         'contact_pairs': dict(pair_counts.most_common()), 'proven_overblocking_examples': false_contacts},
          'summary': {'valid_flat_foot_candidates': sum(r['flat_ground_valid'] for r in scans),
                      'collision_free_candidates': len(valid),
                      'linear_transition_penetrating_frames': sum(bool(r['contacts']) for r in transitions)}}
(OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('RESULT',json.dumps(result['summary']),flush=True)
print('FALSE CONTACT EXAMPLES',len(false_contacts),flush=True)
print('OUTPUT',OUT,flush=True)
