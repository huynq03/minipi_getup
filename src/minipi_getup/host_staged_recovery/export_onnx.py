"""Export the deterministic HoST actor of a checkpoint to ONNX and verify it against PyTorch.

Layer sizes come from the checkpoint weights, so checkpoints without a config.json next to them
(e.g. checkpoints/rev13/model_1100.pt) work too. Writes <out>.onnx and <out>.json (I/O layout,
action scale, hashes).

  python -m minipi_getup.host_staged_recovery.export_onnx --checkpoint checkpoints/rev13/model_1100.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn

from minipi_getup.host.config import PiCfgPPO, class_to_dict
from minipi_getup.host_rl.actor_critic import get_activation
from minipi_getup.host_safe_recovery.asset import REAL_TORQUE
from minipi_getup.host_staged_recovery.asset import ASSET_XML
from minipi_getup.host_staged_recovery.config import StagedRecoveryCfg


def build_actor(state_dict: dict) -> nn.Sequential:
  """Linear layers from the `actor.<i>.weight` tensors, ELU between them, tanh at the end."""
  weights = sorted((int(k.split(".")[1]), v) for k, v in state_dict.items()
                   if k.startswith("actor.") and k.endswith(".weight"))
  activation = class_to_dict(PiCfgPPO())["policy"].get("activation", "elu")
  layers: list[nn.Module] = []
  for i, (_, w) in enumerate(weights):
    layers.append(nn.Linear(w.shape[1], w.shape[0]))
    layers.append(nn.Tanh() if i == len(weights) - 1 else get_activation(activation))
  actor = nn.Sequential(*layers).eval()
  actor.load_state_dict({k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")})
  return actor


def checkpoint_action_scale(saved: dict) -> float | None:
  state = (saved.get("infos") or {}).get("curriculum_state") or {}
  scale = state.get("action_rescale")
  return float(scale.float().mean()) if scale is not None else None


def controller_spec() -> dict:
  """Joint order, PD gains and timing the policy was trained with (for the robot side)."""
  cfg = StagedRecoveryCfg()
  model = mujoco.MjModel.from_xml_path(str(ASSET_XML))
  hinges = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
  names = [model.joint(i).name for i in hinges]
  kind = lambda n: next(k for k in cfg.control.stiffness if n.endswith(f"_{k}_joint"))  # noqa: E731
  return {
    "joint_names": names,
    "joint_range_rad": [model.jnt_range[i].round(4).tolist() for i in hinges],
    "kp": [cfg.control.stiffness[kind(n)] for n in names],
    "kd": [cfg.control.damping[kind(n)] for n in names],
    "torque_limit_nm": [REAL_TORQUE[kind(n)] for n in names],
    "policy_dt_s": cfg.sim.dt * cfg.control.decimation,
    "pd_dt_s": cfg.sim.dt,
    "pd_law": "every pd_dt: target = q_now + action * action_scale; "
              "tau = clip(kp * (target - q_now) - kd * qd_now, +-torque_limit), "
              "i.e. tau = kp * action * action_scale - kd * qd (target re-anchored to the measured q).",
    "imu_frame": "base_link; projected gravity = R_base^T (0, 0, -1)",
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--output", type=Path, help="default: <checkpoint dir>/onnx/<checkpoint stem>.onnx")
  args = p.parse_args()

  ckpt = args.checkpoint.resolve()
  out = args.output or ckpt.parent / "onnx" / f"{ckpt.stem}.onnx"
  saved = torch.load(ckpt, map_location="cpu", weights_only=False)
  actor = build_actor(saved["model_state_dict"])
  num_obs, num_act = actor[0].in_features, actor[-2].out_features

  out.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    actor, torch.zeros(1, num_obs), str(out), input_names=["observations"], output_names=["actions"],
    dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}}, opset_version=18, dynamo=False,
  )
  onnx.checker.check_model(str(out))
  reference = ReferenceEvaluator(str(out))
  gen = torch.Generator().manual_seed(0)
  max_err = 0.0
  for batch in (1, 8, 32):
    for scale in (0.0, 1.0, 10.0, 100.0):
      obs = (torch.randn(batch, num_obs, generator=gen) * scale).clamp(-100, 100)
      with torch.inference_mode():
        expected = actor(obs).numpy()
      actual = reference.run(None, {"observations": obs.numpy()})[0]
      # float32 accumulation over large inputs differs by ~1e-4; far below one action step
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-4)
      max_err = max(max_err, float(np.abs(actual - expected).max()))

  infos = saved.get("infos") or {}
  meta = {
    "source_checkpoint": str(ckpt),
    "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "variant": infos.get("variant"),
    "asset_sha256": infos.get("asset_sha256"),
    "source_commit": infos.get("source_commit"),
    "action_scale": checkpoint_action_scale(saved),
    "input": {"name": "observations", "shape": ["batch", num_obs], "dtype": "float32"},
    "output": {"name": "actions", "shape": ["batch", num_act], "dtype": "float32"},
    "observation": "6 frames, oldest first; each: base angular velocity * 0.25 (3), projected gravity (3), "
                   "joint positions (12), joint velocities * 0.05 (12), previous actions (12), action scale (1); "
                   "clip to [-100, 100] before the model.",
    "action": "tanh output; joint_pos_target = current_joint_pos + action * action_scale (outside the model).",
    "onnx_max_abs_error_vs_torch": max_err,
    "controller": controller_spec(),
  }
  out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
  print(f"exported {out}  [batch, {num_obs}] -> [batch, {num_act}], max |onnx - torch| {max_err:.2e}, "
        f"action scale {meta['action_scale']}")


if __name__ == "__main__":
  main()
