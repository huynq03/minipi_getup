"""Export the deployable student policy and its deploy contract (simulation only).

ONNX: one float32 input ``obs`` [1, 240] = o_{t-4..t} frame-major (oldest first), one
float32 output ``actions`` [1, 12] = raw action (before raw clip). The actor's o_t is
the newest frame, sliced inside the graph, so mini_pi_fsm needs a single observation
group with ``history_length: 5``, ``use_gym_history: true``, ``history_init: zero``.

The runtime action path (raw clip -> scale/offset -> per-joint clip -> PD, torque
clipped at +-16 Nm; experiment pd16_noslew has no target slew) is NOT in the graph.
It is written to ``deploy_contract.json`` for the (not yet implemented) GetUp state.

Usage (artifact):
  python -m minipi_getup.ftsr_ref.export --checkpoint RUN/model_N.pt --out DIR

Never talks to hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import torch

from minipi_getup.ftsr_ref.rl.modules import FtsrModel, StudentPolicy


def sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def load_model(ckpt_path: str) -> FtsrModel:
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  num_actions = ckpt["model"]["std"].shape[0]
  model = FtsrModel(ckpt["dims"], num_actions)
  model.load_state_dict(ckpt["model"])
  return model.eval()


def export_onnx(model: FtsrModel, path: str) -> None:
  policy = StudentPolicy(model).to("cpu").eval()
  dummy = torch.zeros(1, model.dims["policy"])
  torch.onnx.export(
    policy,
    (dummy,),
    path,
    input_names=["obs"],
    output_names=["actions"],
    opset_version=18,
    dynamo=False,
  )


def onnx_parity(model: FtsrModel, path: str, n: int = 64, seed: int = 0) -> float:
  """Max |ONNX - PyTorch| over random inputs (onnx.reference evaluator)."""
  import onnx
  from onnx.reference import ReferenceEvaluator

  sess = ReferenceEvaluator(onnx.load(path))
  policy = StudentPolicy(model).to("cpu").eval()
  g = torch.Generator().manual_seed(seed)
  worst = 0.0
  for _ in range(n):
    x = torch.randn(1, model.dims["policy"], generator=g)
    with torch.no_grad():
      ref = policy(x).numpy()
    out = sess.run(None, {"obs": x.numpy().astype(np.float32)})[0]
    worst = max(worst, float(np.abs(out - ref).max()))
  return worst


def deploy_contract() -> dict:
  """Everything the GetUp runtime must reproduce (policy order)."""
  import re

  from minipi_getup.ftsr_ref.config.env_cfg import (
    ACTION_SCALE,
    PASSIVE_STEPS,
    RAW_CLIP,
  )
  from minipi_getup.ftsr_ref.config.robot import (
    JOINT_NAMES,
    JOINT_RANGES,
    KD,
    KP,
    PASSIVE_KD,
    TAU_CAP,
  )

  def per_joint(table):
    out = []
    for name in JOINT_NAMES:
      hits = [v for k, v in table.items() if re.fullmatch(k, name)]
      assert len(hits) == 1
      out.append(hits[0])
    return out

  short = {k: v for k, v in KP.items()}
  return {
    "joint_names": list(JOINT_NAMES),
    "default_joint_pos": [0.0] * len(JOINT_NAMES),
    "stiffness": [short[n[2:].replace("_joint", "")] for n in JOINT_NAMES],
    "damping": [KD[n[2:].replace("_joint", "")] for n in JOINT_NAMES],
    "step_dt": 0.02,
    "observation": {
      "group": "obs",
      "history_length": 5,
      "use_gym_history": True,
      "history_init": "zero",
      "frame": [
        ["base_lin_vel", "fixed [0, 0, 0]", 3],
        ["base_ang_vel (IMU gyro)", "scale 0.25", 3],
        ["projected_gravity", "scale 1", 3],
        ["velocity_commands (vx, vy, wz)", "scale [2, 2, 0.25]", 3],
        ["joint_pos_rel", "scale 1", 12],
        ["joint_vel", "scale 0.05", 12],
        ["last_action (raw after raw_clip)", "scale 1", 12],
      ],
    },
    "action": {
      "raw_clip": [-RAW_CLIP, RAW_CLIP],
      "scale": per_joint(ACTION_SCALE),
      "offset": [0.0] * len(JOINT_NAMES),
      "clip": [list(JOINT_RANGES[n]) for n in JOINT_NAMES],
      "slew_rad_per_step": None,
      "torque_limit_nm": TAU_CAP,
      "order": "raw_clip -> scale/offset -> clip -> PD (torque clipped +-16 Nm)",
      "note": "trained without a target slew (pd16_noslew); deploying it with a "
      "slew changes the plant",
    },
    "entry": {
      "passive_before_policy_s": PASSIVE_STEPS * 0.02,
      "passive_gains": {"kp": 0.0, "kd": PASSIVE_KD},
    },
    "safety_required": {
      "joint_velocity_fault_rad_s": 4.0,
      "note": "GetUp-state-local; plus stale IMU/motor, NaN, joint/target bounds, "
      "timeout, passive transition. Not implemented in mini_pi_fsm yet.",
    },
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--note", default="")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  model = load_model(args.checkpoint)
  onnx_path = os.path.join(args.out, "policy.onnx")
  export_onnx(model, onnx_path)
  err = onnx_parity(model, onnx_path)
  assert err < 1e-5, err
  shutil.copy(args.checkpoint, os.path.join(args.out, "model.pt"))
  with open(os.path.join(args.out, "deploy_contract.json"), "w") as f:
    json.dump(deploy_contract(), f, indent=2)
  meta = {
    "source_checkpoint": os.path.abspath(args.checkpoint),
    "note": args.note,
    "onnx_parity_max_abs": err,
    "files": {
      f: sha256(os.path.join(args.out, f))
      for f in sorted(os.listdir(args.out))
      if f != "metadata.json"
    },
  }
  with open(os.path.join(args.out, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
  print(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
