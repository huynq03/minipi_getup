"""Staged-recovery asset: CL_PAI/HoST hybrid with 120-degree outward hip roll.

Same file as host_safe_recovery's asset except the hip-roll outward limit
(pi/2 -> 2pi/3); hip pitch stays -1.25..1.75 rad. Kept as a separate file so the
baseline XML and its checkpoints keep their hash.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minipi_getup.host_safe_recovery import asset as safe_asset

ASSET_XML = Path(
  "/home/huy/minipi_getup/src/minipi_getup/asset_zoo/robots/"
  "hightorque_minipi/xmls/cl_pai_host_torso_host_legs_hip120.xml"
)
EXPECTED_ASSET_SHA256 = "45c5882bd0597d161be3a81277c9663ab6c3c5183b56854d9dd919275854f71b"


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
