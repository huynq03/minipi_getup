"""Package a selected FTSR checkpoint as a stable artifact (simulation only).

Creates ``<out>/`` with:
- ``model.pt``: copy of the checkpoint (all networks; the student path is deployable).
- ``policy.onnx``: the student policy ``(o_{t:t-4}, o_t) -> a_t``, checked against
  PyTorch with ``onnx.reference``.
- ``env.yaml`` / ``agent.yaml``: config snapshot from the source run.
- ``metrics.json``: the checkpoint's evaluation row(s) from the results CSV.
- ``metadata.json``: source run, iteration, observation layout, action mapping and
  SHA256 of every file.

Usage:
  python -m minipi_getup.ftsr.export --checkpoint RUN/model_N.pt --out artifacts/ftsr_best \
      --results logs/ftsr_analysis/results.csv --note "why this checkpoint"

This prepares a deployment artifact only. It never talks to hardware.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil

import numpy as np
import torch

from minipi_getup.ftsr.rl.modules import FtsrModel, StudentPolicy


def sha256(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", default="artifacts/ftsr_best")
  ap.add_argument("--results", default="logs/ftsr_analysis/results.csv")
  ap.add_argument("--note", default="")
  args = ap.parse_args()

  ckpt_path = os.path.abspath(args.checkpoint)
  run_dir = os.path.dirname(ckpt_path)
  os.makedirs(args.out, exist_ok=True)
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  dims = ckpt["dims"]
  num_actions = ckpt["model"]["std"].shape[0]
  model = FtsrModel(
    dims["actor"], dims["student"], dims["teacher"], dims["critic"], num_actions
  )
  model.load_state_dict(ckpt["model"])
  policy = StudentPolicy(model).eval()

  shutil.copy2(ckpt_path, os.path.join(args.out, "model.pt"))
  onnx_path = os.path.join(args.out, "policy.onnx")
  dummy = (torch.zeros(1, dims["student"]), torch.zeros(1, dims["actor"]))
  torch.onnx.export(
    policy,
    dummy,
    onnx_path,
    input_names=["student_obs_history", "actor_obs"],
    output_names=["actions"],
    opset_version=18,
    dynamo=False,
  )
  import onnx
  from onnx.reference import ReferenceEvaluator

  onnx.checker.check_model(onnx_path)
  ref = ReferenceEvaluator(onnx_path)
  g = torch.Generator().manual_seed(0)
  hist = torch.randn(8, dims["student"], generator=g)
  obs = torch.randn(8, dims["actor"], generator=g)
  with torch.no_grad():
    want = policy(hist, obs).numpy()
  got = np.concatenate(
    [
      ref.run(
        None,
        {
          "student_obs_history": hist[i : i + 1].numpy(),
          "actor_obs": obs[i : i + 1].numpy(),
        },
      )[0]
      for i in range(8)
    ]
  )
  max_err = float(np.abs(got - want).max())
  assert max_err < 1e-4, f"ONNX mismatch {max_err}"

  for name in ("env.yaml", "agent.yaml"):
    src = os.path.join(run_dir, "params", name)
    if os.path.exists(src):
      shutil.copy2(src, os.path.join(args.out, name))

  rows = []
  if os.path.exists(args.results):
    with open(args.results, newline="") as f:
      for r in csv.DictReader(f):
        if os.path.abspath(r["checkpoint"]) == ckpt_path:
          rows.append(r)
  with open(os.path.join(args.out, "metrics.json"), "w") as f:
    json.dump(rows, f, indent=2)

  meta = {
    "source_checkpoint": ckpt_path,
    "source_run": os.path.basename(run_dir),
    "iteration": ckpt.get("iter"),
    "note": args.note,
    "onnx_max_abs_error_vs_torch": max_err,
    "observation_layout": {
      "actor_obs (o_t)": "ang_vel*0.25 (3), projected_gravity (3), command*(2,2,0.25)"
      " (3), gait_phase [sin,cos] (2), joint_pos - default (12), joint_vel*0.05 (12),"
      " last_action (12)",
      "student_obs_history": "o_{t-4..t}, 5 frames, term-major (each term's 5 frames"
      " consecutive, oldest first), 235 values",
      "joint_order": "r_hip_pitch, r_hip_roll, r_thigh, r_calf, r_ankle_pitch,"
      " r_ankle_roll, l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch,"
      " l_ankle_roll",
    },
    "action": "q_target = q_default (all 0) + 0.25 * clip(a, -7, 7), clamped to the"
    " joint limits, held for the 20 ms control period",
    "control": "50 Hz policy; PD kp/kd = 0.7 x vendor kp, vendor kd;"
    " operational torque envelope 9 Nm (hardware absolute limit 16 Nm)",
    "dims": dims,
    "files": {},
  }
  for name in sorted(os.listdir(args.out)):
    if name != "metadata.json":
      meta["files"][name] = sha256(os.path.join(args.out, name))
  with open(os.path.join(args.out, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
  print(
    json.dumps({k: meta[k] for k in ("source_run", "iteration", "files")}, indent=2)
  )
  print(f"onnx max abs error vs torch: {max_err:.2e}")


if __name__ == "__main__":
  main()
