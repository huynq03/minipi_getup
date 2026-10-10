"""Load the required hybrid MiniPi asset without modifying baseline HoST."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mjlab.actuator import BuiltinMotorActuatorCfg

import minipi_getup.host.env as host_env

ASSET_XML = Path(
  "/home/huy/minipi_getup/src/minipi_getup/asset_zoo/robots/"
  "hightorque_minipi/xmls/cl_pai_host_torso.xml"
)
EXPECTED_ASSET_SHA256 = "cda43fb547589de89adbd740e01e416308265a4ca96661e085699f20193415e9"

REAL_KP = {
  "hip_pitch": 60.0,
  "hip_roll": 40.0,
  "thigh": 20.0,
  "calf": 60.0,
  "ankle_pitch": 30.0,
  "ankle_roll": 10.0,
}
REAL_KD = {
  "hip_pitch": 2.4,
  "hip_roll": 0.8,
  "thigh": 0.4,
  "calf": 2.8,
  "ankle_pitch": 1.6,
  "ankle_roll": 0.3,
}
REAL_TORQUE = {
  "hip_pitch": 16.0,
  "hip_roll": 16.0,
  "thigh": 16.0,
  "calf": 16.0,
  "ankle_pitch": 16.0,
  "ankle_roll": 6.0,
}

_HOST_FACTORY = host_env.get_robot_cfg


def asset_sha256() -> str:
  return hashlib.sha256(ASSET_XML.read_bytes()).hexdigest()


def configure_asset() -> Path:
  """Select the exact requested XML and add only non-physical HoST frames."""
  if not ASSET_XML.is_file():
    raise FileNotFoundError(ASSET_XML)
  digest = asset_sha256()
  if digest != EXPECTED_ASSET_SHA256:
    raise RuntimeError(
      f"Asset changed: expected {EXPECTED_ASSET_SHA256}, found {digest}: {ASSET_XML}"
    )
  host_env.ASSET_XML = ASSET_XML

  def factory(limit_solref=None):
    cfg = _HOST_FACTORY(limit_solref)
    source = cfg.spec_fn

    def spec_fn():
      spec = source()
      base = next(b for b in spec.bodies if b.name == "base_link")
      base.add_body(name="keyframe_head_link", pos=[0, 0, 0.08])
      offsets = ([-0.04, 0.1, 0], [-0.04, -0.1, 0], [0.07, 0, 0], [-0.15, 0, 0])
      for side in ("r", "l"):
        foot = next(b for b in spec.bodies if b.name == f"{side}_ankle_roll_link")
        for i, pos in enumerate(offsets, 1):
          foot.add_body(name=f"auxiliary_{side}_ankle_roll_link{i}", pos=pos)
      return spec

    cfg.spec_fn = spec_fn
    cfg.articulation.actuators = tuple(
      BuiltinMotorActuatorCfg(
        target_names_expr=(f".*_{kind}_joint",),
        effort_limit=limit,
        armature=0.01,
      )
      for kind, limit in REAL_TORQUE.items()
    )
    return cfg

  host_env.get_robot_cfg = factory
  return ASSET_XML
