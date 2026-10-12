"""Staged-recovery asset: CL_PAI/HoST hybrid, 120-degree hip roll, reversible knee, shells.

Baseline XML with the hip-roll outward limit widened (pi/2 -> 2pi/3), the knee
range widened from -0.65..1.65 (-0.65 = straight leg) to -2.5..1.65 so the knee
can bend backwards up to 106 degrees (beyond ~-2.6 the foot would pass through
its own thigh/hip, which no collision stops) (hip pitch stays -1.25..1.75 rad), plus massless box shells fitted to the thigh, shin and foot
mesh AABBs, plus a torso shell copying the torso box. Bits: left 2/12, right 4/10,
torso 8/6, so shells collide left-vs-right and leg-vs-torso only (no terrain or
same-leg contact). Without them the baseline collision model let the two feet
overlap by up to ~2 cm, the shins/thighs pass through each other, and a folded
foot/shin pass into the torso.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minipi_getup.host_safe_recovery import asset as safe_asset

ASSET_XML = Path(
  "/home/huy/minipi_getup/src/minipi_getup/asset_zoo/robots/"
  "hightorque_minipi/xmls/cl_pai_host_torso_host_legs_hip120_legcoll.xml"
)
EXPECTED_ASSET_SHA256 = "be97b3280fedcb710f971b5bfcca7e9433a2e21d049b67ec064d21d3cbffe531"


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
