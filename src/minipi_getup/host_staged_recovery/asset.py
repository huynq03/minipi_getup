"""Staged-recovery asset: CL_PAI/HoST hybrid, 120-degree hip roll, leg shells.

Baseline XML with the hip-roll outward limit widened (pi/2 -> 2pi/3; hip pitch
stays -1.25..1.75 rad) plus massless box shells fitted to the thigh, shin and foot
mesh AABBs. The shells use bits l=2/4, r=4/2, so they collide only left-vs-right:
no terrain, torso, or same-leg contact. Without them the baseline collision model
let the two feet overlap by up to ~2 cm and the shins/thighs pass through each other.
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
EXPECTED_ASSET_SHA256 = "eba62848d93dd88a50ee9ac43191b08baa8a67b483cef1be017f411da776f0c8"


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
