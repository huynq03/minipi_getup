"""Staged-recovery asset: CL_PAI/HoST hybrid, 120-degree hip roll, reversible knee, shells.

Baseline XML with the hip-roll outward limit widened (pi/2 -> 2pi/3), the knee
range widened from -0.65..1.65 (-0.65 = straight leg) to -2.5..1.65 (106 degrees
backwards, the last angle where the foot clears the torso and hip motor with the
ankle neutral); thigh yaw toward the other leg capped at 0.28 (the feet meet at
0.30 when both legs turn heels-in); hip-motor shells make the same leg's foot
collide with its hip, so a reversed knee with the toe pointed down is stopped
physically (hip pitch stays -1.25..1.75 rad), plus massless box shells fitted to the thigh, shin and foot
mesh AABBs, plus a torso shell copying the torso box. Bits: left 2/12, right 4/10,
torso 8/6, foot shells also carry 64 (l) / 128 (r) for the hip shells
(16/64 l, 32/128 r), so shells collide left-vs-right, leg-vs-torso and
foot-vs-own-hip only (no terrain or
same-leg contact). Without them the baseline collision model let the two feet
overlap by up to ~2 cm, the shins/thighs pass through each other, and a folded
foot/shin pass into the torso.

thigh1 (2026-10-16): thigh yaw -1..1 rad on both legs (was -0.28..0.30 right, -0.30..0.28
left). The leg shells now stop the feet from crossing, and the HoST URDF allows +-3 rad.
Otherwise identical to cl_pai_host_torso_host_legs_hip120_legcoll.xml (sha256
ea7d909d...), which model_1100 / getup_2600 / ft2600 were trained on.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minipi_getup.host_safe_recovery import asset as safe_asset

# Package-relative, so a fresh clone of this branch finds it.
ASSET_XML = (
  Path(__file__).resolve().parents[1]
  / "asset_zoo/robots/hightorque_minipi/xmls/cl_pai_host_torso_host_legs_hip120_thigh1_legcoll.xml"
)
EXPECTED_ASSET_SHA256 = "7bf7d894bda3831db43e5d14bc9006412c2802afb9e3bc3c2b9b87f673562a4c"


def asset_sha256() -> str:
  return safe_asset.asset_sha256(ASSET_XML)


def configure_asset() -> Path:
  return safe_asset.configure_asset(ASSET_XML, EXPECTED_ASSET_SHA256)


def asset_manifest() -> dict[str, str]:
  """SHA256 of the XML and every mesh it references."""
  text = ASSET_XML.read_text()
  meshdir = ASSET_XML.parent / re.search(r'meshdir="([^"]+)"', text).group(1)
  out = {ASSET_XML.name: asset_sha256()}
  for name in sorted(set(re.findall(r'file="([^"]+)"', text))):
    out[name] = hashlib.sha256((meshdir / name).read_bytes()).hexdigest()
  return out
