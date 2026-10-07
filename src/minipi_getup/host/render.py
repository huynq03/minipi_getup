"""Offscreen MuJoCo rendering of recorded HoST-port rollouts (qpos per policy step)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


class Renderer:
  def __init__(self, mj_model: mujoco.MjModel, width=640, height=480):
    self.m = mj_model
    self.d = mujoco.MjData(mj_model)
    self.r = mujoco.Renderer(mj_model, height=height, width=width)
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.distance = 1.3
    self.cam.elevation = -12.0
    self.cam.azimuth = 120.0
    self.opt = mujoco.MjvOption()

  def frame(self, qpos: np.ndarray, lookat=None) -> np.ndarray:
    self.d.qpos[:] = qpos
    mujoco.mj_forward(self.m, self.d)
    self.cam.lookat[:] = lookat if lookat is not None else [qpos[0], qpos[1], 0.2]
    self.r.update_scene(self.d, self.cam, self.opt)
    return self.r.render()

  @staticmethod
  def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
      dr.text((8, 8 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def write_video(path: str, frames: list[np.ndarray], fps: int = 50) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, quality=8)
