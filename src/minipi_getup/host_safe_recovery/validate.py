"""CPU validation for the isolated HoST safe-recovery variant."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from minipi_getup.host_safe_recovery.asset import (
  ASSET_XML,
  EXPECTED_ASSET_SHA256,
  REAL_KD,
  REAL_KP,
  REAL_TORQUE,
)

BASELINE_HASHES = {
  "src/minipi_getup/host/config.py": "d97a7da1ee939b6e7fe78936f64495a88e55a37d8539749528620d5c1e07122c",
  "src/minipi_getup/host/env.py": "67d08c39e6c2c2bbc59c26e5f81d39568b1a4150b1f4d787e05aeb0dbf5e80b2",
  "src/minipi_getup/host_rl/actor_critic.py": "3ea8307d29a486828d6a914a2fe6cb72116dab15e1aabcea826a43d739d6e248",
  "src/minipi_getup/host_rl/on_policy_runner.py": "591bef8cc3bcfb5c80150e8c658c3a9fc8ad3df4ecfdc3a31d22926b86b61c1c",
  "src/minipi_getup/host_rl/ppo.py": "d4b517ee25f156b2b82957e447d2680ffe5b0abc10d800e73539e7d3cacbe4d9",
  "src/minipi_getup/host_rl/rollout_storage.py": "f82753abafbceff3dbc9b83ff6bb046aa763a10dad17f8334e67a754a9e70101",
}


def sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  root = Path(__file__).resolve().parents[3]
  assert sha(ASSET_XML) == EXPECTED_ASSET_SHA256
  xml = ET.parse(ASSET_XML).getroot()
  compiler = xml.find("compiler")
  meshdir = ASSET_XML.parent / compiler.attrib["meshdir"]
  for mesh in xml.findall("./asset/mesh"):
    assert (meshdir / mesh.attrib["file"]).is_file(), mesh.attrib["file"]

  model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
  joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
  assert len(joints) == 12 and model.nq == 19 and model.nv == 18
  assert np.all(model.jnt_range[1:, 0] < model.jnt_range[1:, 1])
  assert set(REAL_KP) == set(REAL_KD) == set(REAL_TORQUE)
  assert REAL_TORQUE["ankle_roll"] == 6 and all(
    v == 16 for k, v in REAL_TORQUE.items() if k != "ankle_roll"
  )
  assert not any("keyframe" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "") for i in range(model.nbody))

  for rel, expected in BASELINE_HASHES.items():
    assert sha(root / rel) == expected, f"baseline HoST changed: {rel}"
  print("PASS: exact asset and all meshes load")
  print("PASS: 12 joints, real XML ROM, real PD map, 16/6 Nm torque map")
  print("PASS: original HoST environment, PPO, actor/critics and storage unchanged")


if __name__ == "__main__":
  main()
