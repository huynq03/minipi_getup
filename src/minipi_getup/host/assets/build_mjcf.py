"""Generate ``pi_12dof_host.xml`` from HoST's ``pi_12dof_release_v1.urdf``.

The URDF is copied verbatim from ``HoST/legged_gym/resources/robots/pi_12dof``. The
conversion is mechanical so that the MuJoCo model has the same kinematics, inertials,
joint limits and collision shapes Isaac Gym loaded:

- URDF joint origins (xyz + fixed-axis rpy) become body pos/quat.
- ``<inertial>`` becomes ``fullinertia`` (no rebalancing).
- Revolute joints become hinges with the URDF range. Armature is 0.01, the value of
  ``PiCfg.asset.armature`` that Isaac Gym applied to every DOF.
- ``<collision>`` boxes/spheres/meshes are kept as is (MuJoCo, like PhysX, collides
  meshes through their convex hull). ``<visual>`` meshes are non-colliding.
- Fixed ``dont_collapse`` links (``keyframe_head_link``, ``auxiliary_*``) are kept as
  massless welded bodies, so their frame origins are available like Isaac Gym's
  rigid-body states.
- Meshes resolve to ``asset_zoo/robots/hightorque_minipi/xmls/assets``; every mesh the
  model uses is byte-identical to HoST's ``pi_12dof/meshes`` (its ``base_link.STL`` is
  referenced only by a commented-out collision and is not needed).

Run: ``python -m minipi_getup.host.assets.build_mjcf``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
URDF = HERE / "pi_12dof_release_v1.urdf"
OUT = HERE / "pi_12dof_host.xml"
ARMATURE = 0.01


def _f(s: str | None, n: int = 3) -> list[float]:
  return [float(v) for v in s.split()] if s else [0.0] * n


def _rpy_to_quat(r: float, p: float, y: float) -> list[float]:
  # URDF rpy: R = Rz(y) Ry(p) Rx(r). Returned as MuJoCo (w, x, y, z).
  cr, sr = math.cos(r / 2), math.sin(r / 2)
  cp, sp = math.cos(p / 2), math.sin(p / 2)
  cy, sy = math.cos(y / 2), math.sin(y / 2)
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ]


def _fmt(v: list[float]) -> str:
  return " ".join(f"{x:.10g}" for x in v)


def build() -> str:
  root = ET.parse(URDF).getroot()
  links = {link.get("name"): link for link in root.findall("link")}
  joints = root.findall("joint")
  children: dict[str, list[ET.Element]] = {}
  for j in joints:
    children.setdefault(j.find("parent").get("link"), []).append(j)
  child_links = {j.find("child").get("link") for j in joints}
  (base,) = [n for n in links if n not in child_links]

  meshes: set[str] = set()
  out: list[str] = []

  def geom_xml(el: ET.Element, visual: bool) -> str | None:
    origin = el.find("origin")
    pos = _f(origin.get("xyz")) if origin is not None else [0.0] * 3
    rpy = _f(origin.get("rpy")) if origin is not None else [0.0] * 3
    g = el.find("geometry")
    attrs = f'pos="{_fmt(pos)}"'
    if any(abs(a) > 0 for a in rpy):
      attrs += f' quat="{_fmt(_rpy_to_quat(*rpy))}"'
    if (m := g.find("mesh")) is not None:
      name = Path(m.get("filename")).stem
      meshes.add(name)
      shape = f'type="mesh" mesh="{name}"'
    elif (b := g.find("box")) is not None:
      shape = f'type="box" size="{_fmt([s / 2 for s in _f(b.get("size"))])}"'
    elif (s := g.find("sphere")) is not None:
      shape = f'type="sphere" size="{s.get("radius")}"'
    else:
      raise ValueError(f"unsupported geometry in {ET.tostring(g)}")
    cls = "visual" if visual else "collision"
    if visual and "mesh" not in shape:
      return None  # Keyframe/auxiliary marker shapes are visual-only decorations.
    return f'<geom class="{cls}" {shape} {attrs}/>'

  def emit_link(name: str, joint: ET.Element | None, depth: int) -> None:
    ind = "  " * depth
    link = links[name]
    attrs = f'name="{name}"'
    if joint is not None:
      o = joint.find("origin")
      attrs += f' pos="{_fmt(_f(o.get("xyz")))}"'
      rpy = _f(o.get("rpy"))
      if any(abs(a) > 0 for a in rpy):
        attrs += f' quat="{_fmt(_rpy_to_quat(*rpy))}"'
    out.append(f"{ind}<body {attrs}>")
    inertial = link.find("inertial")
    if inertial is not None:
      o = inertial.find("origin")
      i = inertial.find("inertia")
      full = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
      out.append(
        f'{ind}  <inertial pos="{_fmt(_f(o.get("xyz")))}" '
        f'mass="{inertial.find("mass").get("value")}" fullinertia="{_fmt(full)}"/>'
      )
    if joint is None:
      out.append(f'{ind}  <freejoint name="floating_base"/>')
    elif joint.get("type") == "revolute":
      lim = joint.find("limit")
      out.append(
        f'{ind}  <joint name="{joint.get("name")}" type="hinge" '
        f'axis="{_fmt(_f(joint.find("axis").get("xyz")))}" '
        f'range="{lim.get("lower")} {lim.get("upper")}" armature="{ARMATURE}"/>'
      )
    elif joint.get("type") != "fixed":
      raise ValueError(joint.get("type"))
    for v in link.findall("visual"):
      if (g := geom_xml(v, True)) is not None:
        out.append(f"{ind}  {g}")
    for k, c in enumerate(link.findall("collision")):
      g = geom_xml(c, False)
      out.append(f"{ind}  {g.replace('<geom ', f'<geom name="{name}_col{k}" ', 1)}")
    for cj in children.get(name, []):
      emit_link(cj.find("child").get("link"), cj, depth + 1)
    out.append(f"{ind}</body>")

  emit_link(base, None, 2)
  mesh_xml = "\n".join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in sorted(meshes))
  body_xml = "\n".join(out)
  return f"""<!-- Generated by build_mjcf.py from HoST pi_12dof_release_v1.urdf. Do not edit. -->
<mujoco model="pi_12dof_host">
  <compiler angle="radian" meshdir="../../asset_zoo/robots/hightorque_minipi/xmls/assets" autolimits="true"/>
  <default>
    <default class="visual">
      <geom group="2" contype="0" conaffinity="0" density="0" rgba="0.75 0.75 0.75 1"/>
    </default>
    <default class="collision">
      <geom group="3" contype="1" conaffinity="1" condim="3" rgba=".2 .6 .2 .4"/>
    </default>
  </default>
  <asset>
{mesh_xml}
  </asset>
  <worldbody>
{body_xml}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
  OUT.write_text(build())
  print(f"wrote {OUT}")
