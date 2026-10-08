"""Visual demo of a recovery checkpoint (simulation only, evaluation tooling; plant PD16, no slew).

Runs the same deterministic student rollout as ``evaluate.py`` (play cfg: no noise,
poses by env index, 2 s passive + 18 s actuated), once without assistance and once
with the Eq. 4 assistance frozen at ``tc``, for a fixed env seed. Records every
env's state per policy step, then renders selected envs offline with MuJoCo:

- per pose, the first env of that pose ("selected rollout", whatever its outcome);
- without assistance, if the selected env failed, also the lowest-index env of that
  pose that recovered ("example success").

Outcome per env, from the evaluation's own quantities: RECOVERED if, over the last
3 s, the mean base height is above h1 (0.19 m) and the mean uprightness above
cos 18 deg (``frac_envs_final_above_h1`` and ``frac_envs_final_upright``). This is
recovery to foot support; the stricter ``success`` metric (0.31 m) is reported too.

Usage:
  python -m minipi_getup.ftsr_ref.visual_demo --checkpoint RUN/model_N.pt \
      --out results/model_N_visual [--seed 2150] [--tc 0.2]
"""

from __future__ import annotations

import argparse
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import minipi_getup  # noqa: F401
from minipi_getup.ftsr_ref.config.env_cfg import STAGE_HEIGHTS
from minipi_getup.ftsr_ref.config.robot import MINIPI_MASS
from minipi_getup.ftsr_ref.evaluate import (
  DEV,
  STAND_H,
  STAND_UP,
  SupportTracker,
  build_env,
)
from minipi_getup.ftsr_ref.export import load_model
from minipi_getup.ftsr_ref.mdp.resets import FALLEN_POSE_NAMES, POSE_KEY
from minipi_getup.ftsr_ref.rl.modules import StudentPolicy

TASK = "Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless"  # v2 semantics (model_2150's run)
W, H = 800, 600
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def rollout(model, per_pose: int, seed: int, tc: float | None) -> dict:
  """Deterministic student rollout of ``per_pose`` envs per pose; returns numpy
  arrays per policy step (after each env step)."""
  npose = len(FALLEN_POSE_NAMES)
  n = per_pose * npose

  def mutate(cfg):
    cfg.events["reset_pose"].params["by_env_index"] = True
    cfg.seed = seed

  torch.manual_seed(seed)
  env = build_env(TASK, n, mutate, tc)
  t = env.action_manager.get_term("joint_pos")
  pose = env.extras[POSE_KEY].cpu().numpy()
  support = SupportTracker(env)
  policy = StudentPolicy(model).to(DEV).eval()
  robot = env.scene["robot"]
  steps = int(env.max_episode_length) - 1
  qpos, h, up, fz, sup, hcmd = [], [], [], [], [], []
  stage = env.extras["ftsr_stage"]
  origins = env.scene.env_origins.cpu().numpy()
  q0 = env.sim.data.qpos.clone()
  obs = env.get_observations()
  with torch.no_grad():
    for _ in range(steps):
      obs, *_ = env.step(policy(obs["policy"]))
      d = robot.data
      qpos.append(env.sim.data.qpos.cpu().numpy().copy())
      h.append((d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy())
      up.append((-d.projected_gravity_b[:, 2]).cpu().numpy())
      # Eq. 4 wrench applied in the last physics step (world z); none -> 0.
      f = getattr(t, "force", None) if t.cfg.assist is not None else None
      fz.append(np.zeros(n) if f is None else f[:, 2].cpu().numpy().copy())
      sup.append(support.touching().cpu().numpy())
      hcmd.append((stage.stage, stage.h_cmd))
  out = {
    "pose": pose,
    "origins": origins,
    "qpos0": q0.cpu().numpy(),
    "qpos": np.stack(qpos, 1),  # (n, T, nq)
    "h": np.stack(h, 1),
    "up": np.stack(up, 1),
    "fz": np.stack(fz, 1),
    "support": np.stack(sup, 1),  # (n, T, groups)
    "stage": hcmd,  # population stage and Eq. 4 h_cmd per step
    "passive": t.cfg.passive_steps,
    "dt": env.step_dt,
    "mj_model": env.sim.mj_model,
  }
  env.close()
  return out


def outcome(r: dict) -> dict:
  last = int(round(3.0 / r["dt"]))
  hf = r["h"][:, -last:].mean(1)
  uf = r["up"][:, -last:].mean(1)
  recovered = (hf > STAGE_HEIGHTS[0]) & (uf > STAND_UP)
  strict = ((r["h"][:, -last:] > STAND_H) & (r["up"][:, -last:] > STAND_UP)).all(1)
  feet_only = r["support"][:, -last:, 0].all(1) & ~r["support"][:, -last:, 1:].any(
    (1, 2)
  )
  return {
    "h_final": hf,
    "up_final": uf,
    "recovered": recovered,
    "strict": strict,
    "feet_only": feet_only,
  }


def lying_force(r: dict, i: int) -> float:
  """Mean applied assist force over the first actuated 0.2 s (robot still lying)."""
  p = r["passive"]
  return float(np.abs(r["fz"][i, p : p + 10]).mean())


class Renderer:
  def __init__(self, mj_model):
    self.m = mj_model
    self.d = mujoco.MjData(mj_model)
    self.r = mujoco.Renderer(mj_model, H, W)
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.distance = 1.1
    self.cam.elevation = -18.0
    self.cam.azimuth = 135.0
    self.font = ImageFont.truetype(FONT, 17)
    self.big = ImageFont.truetype(FONT_BOLD, 64)
    self.lookat = None

  def frame(self, qpos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    q = qpos.copy()
    q[0:2] -= origin[0:2]
    self.d.qpos[:] = q
    mujoco.mj_forward(self.m, self.d)
    target = np.array([q[0], q[1], 0.15])
    self.lookat = target if self.lookat is None else 0.9 * self.lookat + 0.1 * target
    self.cam.lookat[:] = self.lookat
    self.r.update_scene(self.d, self.cam)
    return self.r.render()

  def overlay(self, img: np.ndarray, lines: list[str], banner: str | None = None):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    pad, lh = 8, 21
    width = max(dr.textlength(x, font=self.font) for x in lines) + 2 * pad
    dr.rectangle((0, 0, width, pad * 2 + lh * len(lines)), fill=(0, 0, 0, 150))
    for k, s in enumerate(lines):
      dr.text((pad, pad + k * lh), s, font=self.font, fill=(255, 255, 255, 255))
    if banner is not None:
      ok = banner.startswith("RECOVERED")
      color = (40, 200, 90, 230) if ok else (220, 60, 60, 230)
      word = banner.split("\n")[0]
      tw = dr.textlength(word, font=self.big)
      dr.rectangle((0, H // 2 - 60, W, H // 2 + 70), fill=(0, 0, 0, 170))
      dr.text(((W - tw) / 2, H // 2 - 50), word, font=self.big, fill=color)
      for k, s in enumerate(banner.split("\n")[1:]):
        sw = dr.textlength(s, font=self.font)
        dr.text(((W - sw) / 2, H // 2 + 22 + k * lh), s, font=self.font, fill="white")
    return np.asarray(im)


def render_env(rend, r, o, i, header, path_mp4, path_img_prefix) -> dict:
  dt = r["dt"]
  T = r["qpos"].shape[1]
  ok = bool(o["recovered"][i])
  rend.lookat = None
  frames = []
  imgs = {}
  i_init = r["passive"] - 1
  i_high = int(np.argmax(r["h"][i])) if not ok else T - 1
  banner = (
    "RECOVERED\nlast 3 s: base > 0.19 m (h1) and tilt < 18 deg"
    if ok
    else "FAILED\nnot (last 3 s: base > 0.19 m and tilt < 18 deg)"
  )
  for k in range(T):
    img = rend.frame(r["qpos"][i, k], r["origins"][i])
    phase = "passive (kp 0, kd 1)" if k < r["passive"] else "policy"
    lines = header + [
      f"t = {(k + 1) * dt:5.2f} s  [{phase}]",
      f"base height {r['h'][i, k]:.3f} m   assist Fz {r['fz'][i, k]:5.1f} N",
      f"population stage {('r_u', 'r_s', 'r_w')[r['stage'][k][0]]}"
      f"   assist h_cmd {r['stage'][k][1]:.3f} m",
    ]
    frame = rend.overlay(img, lines)
    frames.append(frame)
    if k == i_init:
      imgs["initial"] = frame
    if k == i_high:
      imgs["highest"] = frame
  end = rend.overlay(frames[-1][..., :3].copy(), header, banner)
  frames.extend([end] * int(2.0 / dt))
  imageio.mimwrite(path_mp4, frames, fps=int(round(1.0 / dt)), quality=8)
  for tag, im in imgs.items():
    name = "recovered" if (tag == "highest" and ok) else tag
    imageio.imwrite(f"{path_img_prefix}_{name}.png", im)
  return {
    "video": path_mp4,
    "recovered": ok,
    "strict_success_0p31m": bool(o["strict"][i]),
    "feet_only_last_3s": bool(o["feet_only"][i]),
    "final_height_m": round(float(o["h_final"][i]), 4),
    "final_upright_cos": round(float(o["up_final"][i]), 4),
    "max_height_m": round(float(r["h"][i].max()), 4),
    "assist_mean_last_3s_N": round(
      float(np.abs(r["fz"][i, -int(round(3.0 / dt)) :]).mean()), 2
    ),
    "population_stage_end": ("r_u", "r_s", "r_w")[r["stage"][-1][0]],
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=2150)
  ap.add_argument("--tc", type=float, default=0.2)
  ap.add_argument("--per-pose", type=int, default=64)
  ap.add_argument("--poses", default="", help="comma list; default all four")
  ap.add_argument("--no-assist-only", action="store_true")
  ap.add_argument("--record-name", default="demo_record.json")
  args = ap.parse_args()
  ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
  model = load_model(args.checkpoint).to(DEV)
  tc_tag = f"assist_tc_{str(args.tc).replace('.', 'p')}"
  os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
  record = {
    "checkpoint": os.path.abspath(args.checkpoint),
    "task": TASK,
    "policy": "student (deterministic mean action)",
    "seed": args.seed,
    "envs_per_pose": args.per_pose,
    "env_index_rule": "pose = env_index % 4 (" + ", ".join(FALLEN_POSE_NAMES) + ")",
    "recovered_criterion": "last 3 s mean base height > 0.19 m and mean upright cos"
    f" > {STAND_UP:.4f} (18 deg)",
    "runs": {},
  }
  mg = MINIPI_MASS * 9.81
  poses = set(args.poses.split(",")) if args.poses else set(FALLEN_POSE_NAMES)
  runs = [("no_assist", None)] + ([] if args.no_assist_only else [(tc_tag, args.tc)])
  for name, tc in runs:
    vdir = os.path.join(args.out, "videos", name)
    os.makedirs(vdir, exist_ok=True)
    r = rollout(model, args.per_pose, args.seed, tc)
    o = outcome(r)
    rend = Renderer(r["mj_model"])
    run = {"tc": tc, "per_pose": {}}
    for g, pose in enumerate(FALLEN_POSE_NAMES):
      if pose not in poses:
        continue
      rows = np.nonzero(r["pose"] == g)[0]
      rate = float(o["recovered"][rows].mean())
      chosen = [("selected", int(rows[0]))]
      if tc is None and not o["recovered"][rows[0]]:
        wins = rows[o["recovered"][rows]]
        if len(wins):
          chosen.append(("example_success", int(wins[0])))
      entry = {
        "recovered_rate_this_seed": rate,
        "n": len(rows),
        "videos": [],
      }
      for tag, i in chosen:
        f_lying = lying_force(r, i)
        assist = (
          "none (tc = 0, 0 N)"
          if tc is None
          else f"tc = {tc}, ~{f_lying:.0f} N while lying ({f_lying / mg:.2f} mg)"
        )
        header = [
          "Mini-Pi FTSR",
          f"checkpoint: {ckpt_name}",
          "student policy (deterministic)",
          f"pose: {pose}"
          + ("   [example success]" if tag == "example_success" else ""),
          f"assist: {assist}",
          f"seed: {args.seed}  env index: {i}",
        ]
        base = f"{pose}_{tag}_seed{args.seed}_env{i}"
        res = render_env(
          rend,
          r,
          o,
          i,
          header,
          os.path.join(vdir, base + ".mp4"),
          os.path.join(args.out, "images", f"{name}_{base}"),
        )
        res.update({"role": tag, "env_index": i, "assist_lying_N": round(f_lying, 2)})
        entry["videos"].append(res)
        print(f"[demo] {name} {pose} {tag} env {i}: {res}", flush=True)
      run["per_pose"][pose] = entry
    record["runs"][name] = run
  with open(os.path.join(args.out, args.record_name), "w") as f:
    json.dump(record, f, indent=1)
  print(json.dumps(record, indent=1))


if __name__ == "__main__":
  main()
