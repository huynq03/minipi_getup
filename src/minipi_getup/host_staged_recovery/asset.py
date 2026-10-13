"""Staged-recovery Mini-Pi v2 asset with symmetric hardware-based joint limits.

Uses cl_pai_v2.xml with the original hybrid collision geoms; extra shells removed.
Hip pitch: [-2.28, 2.83]; thigh yaw: [-2.45, 2.45]; calf: [-2.50, 1.39];
ankle pitch: [-0.43, 1.30]; ankle roll: [-0.70, 0.70] radians on both legs.
Hip roll is mirrored: left [-0.13, 2.71], right [-2.71, 0.13].
The asset hash pins the exact model used by new training runs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from minipi_getup.host_safe_recovery import asset as safe_asset

# Package-relative, so a fresh clone of this branch finds it.
ASSET_XML = (
  Path(__file__).resolve().parents[1]
  / "asset_zoo/robots/hightorque_minipi/xmls/cl_pai_v2.xml"
)
EXPECTED_ASSET_SHA256 = "0ab0e6c4c799273e9fa17d3346ea4907ff9c6517e52763e737e56f10967c6558"


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
