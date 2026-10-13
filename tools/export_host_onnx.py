"""Export the deterministic HoST actor and verify ONNX against PyTorch.

Usage: python tools/export_host_onnx.py --checkpoint RUN/model_2600.pt
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from minipi_getup.host_rl.actor_critic import get_activation
from onnx.reference import ReferenceEvaluator
from torch import nn


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  checkpoint = args.checkpoint.resolve()
  output = args.output or checkpoint.parent / "exported" / f"{checkpoint.stem}.onnx"
  config = json.loads((checkpoint.parent / "config.json").read_text())
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  state = {
    key.removeprefix("actor."): value
    for key, value in saved["model_state_dict"].items()
    if key.startswith("actor.")
  }
  policy_cfg = config["train_cfg"]["policy"]
  dims = [config["env_cfg"]["env"]["num_observations"]]
  dims += policy_cfg["actor_hidden_dims"]
  dims += [config["env_cfg"]["env"]["num_actions"]]
  layers = []
  for index, (num_in, num_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
    layers.append(nn.Linear(num_in, num_out))
    layers.append(
      nn.Tanh() if index == len(dims) - 2
      else get_activation(policy_cfg.get("activation", "elu"))
    )
  actor = nn.Sequential(*layers).eval()
  actor.load_state_dict(state, strict=True)
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    actor, torch.zeros(1, dims[0]), str(output),
    input_names=["observations"], output_names=["actions"],
    dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=18, dynamo=False,
  )
  onnx.checker.check_model(str(output))
  reference = ReferenceEvaluator(str(output))
  generator = torch.Generator().manual_seed(0)
  max_error = 0.0
  for batch in (1, 8, 32):
    for scale in (0.0, 1.0, 10.0, 100.0):
      observations = (torch.randn(batch, dims[0], generator=generator) * scale).clamp(-100, 100)
      with torch.inference_mode():
        expected = actor(observations).numpy()
      actual = reference.run(None, {"observations": observations.numpy()})[0]
      np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
      max_error = max(max_error, float(np.abs(actual - expected).max()))
  metadata = {
    "source_checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "iteration": saved.get("it", saved.get("iter")),
    "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "input": {"name": "observations", "shape": ["batch", dims[0]], "dtype": "float32"},
    "output": {"name": "actions", "shape": ["batch", dims[-1]], "dtype": "float32"},
    "observation_history": "6 frames, oldest first; each frame: angular velocity * 0.25 (3), projected gravity (3), joint positions (12), joint velocities * 0.05 (12), previous actions (12), action scale (1). Inputs clipped to [-100, 100] externally.",
    "action_mapping": "Deterministic actor with final tanh. Environment applies joint_pos_target = current_joint_pos + actions * action_rescale; scaling and control are external to ONNX.",
    "checkpoint_action_scale_range": [
      float(saved["infos"]["curriculum_state"]["action_rescale"].min()),
      float(saved["infos"]["curriculum_state"]["action_rescale"].max()),
    ],
    "onnx_max_abs_error_vs_torch": max_error,
    "verification": "ONNX checker and 12 numerical comparisons: batches 1/8/32, input scales 0/1/10/100, seed 0.",
    "training_config": config,
  }
  output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
  print(f"Exported: {output}")
  print(f"Input: [batch, {dims[0]}]; output: [batch, {dims[-1]}]")
  print(f"ONNX max absolute error vs PyTorch: {max_error:.3e}")


if __name__ == "__main__":
  main()
