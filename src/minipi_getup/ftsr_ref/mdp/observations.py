"""Observations of the FTSR reference port (FTSR_REFERENCE_AUDIT_V2.md Sec. 3).

Groups (Mini-Pi dims):

- ``policy`` (240): student history o_{t-4..t}, H = 5 *distinct* frames (release bug
  Q2 fixed), frame-major, oldest first, zero-initialized at reset. The actor's o_t is
  the newest frame (the last 48 values), the same noisy sample, as in the release
  where the history stacks ``obs_buf``. One ONNX input, so any per-evaluation clock
  on the robot would advance once per step.
- ``teacher`` (20): x_t.
- ``critic`` (50): privileged state; the runner prepends the teacher latent.

o_t (48), release order and scales (``compute_observations_without_noise``):

    [0:3)   base_lin_vel * 2 * 0   zero slot (deploy: fixed [0, 0, 0])
    [3:6)   IMU gyro * 0.25
    [6:9)   projected gravity
    [9:12)  command (vx, vy, wz) * (2, 2, 0.25)
    [12:24) q - q_default
    [24:36) qdot * 0.05
    [36:48) last action (raw-clipped, before slew)

Noise (uniform, observation units, release ``noise_scales`` x ``obs_scales``): gyro
0.05 x 0.25, gravity 0.05, q 0.04, qdot 0.06 x 0.05; none on command / action.

Passive window (recovery task): every group is zero while the next step is passive,
like the release's unactuated-window masking (Q14). Here it applies from iteration 0
(the release starts at iteration 500): the deploy GetUp state starts with a
zero-initialized history (``history_init: zero``) and the policy does not run in
Passive, so the first actuated history is [0, 0, 0, 0, o_t] in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

from minipi_getup.ftsr_ref.config.robot import LENGTH_RATIO, MINIPI_WEIGHT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FRAME_DIM = 48
HISTORY = 5
_FRAME_KEY = "ftsr_obs_frame"

# (start, end, noise amplitude) in observation units.
_NOISE = ((3, 6, 0.05 * 0.25), (6, 9, 0.05), (12, 24, 0.04), (24, 36, 0.06 * 0.05))


def _action(env: ManagerBasedRlEnv):
  return env.action_manager.get_term("joint_pos")


def active_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(N, 1) 0.0 while the next env step is passive, else 1.0."""
  act = _action(env)
  passive_next = (env.episode_length_buf < act.cfg.passive_steps) & ~act.entered
  return (~passive_next).float().unsqueeze(-1)


def _frame(env: ManagerBasedRlEnv, true_lin_vel: bool) -> torch.Tensor:
  robot = env.scene["robot"]
  data = robot.data
  act = _action(env)
  ids = act._ids
  cmd = env.command_manager.get_command("twist")
  assert cmd is not None
  lin = (
    data.root_link_lin_vel_b * 2.0
    if true_lin_vel
    else torch.zeros_like(data.root_link_lin_vel_b)
  )
  return torch.cat(
    (
      lin,
      imu_gyro(env) * 0.25,
      data.projected_gravity_b,
      cmd[:, :3] * torch.tensor((2.0, 2.0, 0.25), device=env.device),
      data.joint_pos[:, ids] - act.default,
      data.joint_vel[:, ids] * 0.05,
      act.raw_action,
    ),
    dim=-1,
  )


def imu_gyro(env: ManagerBasedRlEnv) -> torch.Tensor:
  """IMU gyro (site ``imu`` on base_link, identity mount), body frame, rad/s."""
  sensor = env.scene["robot/imu_ang_vel"]
  return sensor.data


def ftsr_obs_frame(env: ManagerBasedRlEnv, noisy: bool = True) -> torch.Tensor:
  """o_t (48), noisy for training, masked in the passive window; cached per step."""
  o = _frame(env, true_lin_vel=False)
  if noisy:
    for lo, hi, amp in _NOISE:
      o[:, lo:hi] += (2.0 * torch.rand_like(o[:, lo:hi]) - 1.0) * amp
  o = o * active_mask(env)
  env.extras[_FRAME_KEY] = o
  return o


class ftsr_history:
  """o_{t-4..t}, frame-major, oldest first; zero-initialized at reset.

  Must be in a group computed after the ``ftsr_obs_frame`` term (same step).
  Pushes one frame per env step (``common_step_counter``); a second evaluation in
  the same step (explicit reset path) replaces the newest frame.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._buf = torch.zeros(env.num_envs, HISTORY, FRAME_DIM, device=env.device)
    self._step = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._buf[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    frame = env.extras[_FRAME_KEY]
    if env.common_step_counter != self._step:
      self._step = env.common_step_counter
      self._buf = torch.roll(self._buf, shifts=-1, dims=1)
    self._buf[:, -1] = frame
    return self._buf.reshape(env.num_envs, -1)


def _base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = env.scene["robot"].data
  body = env.scene["robot"].find_bodies(("base_link",))[0][0]
  return data.body_link_pos_w[:, body, 2:3]


def ftsr_teacher_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """x_t (20), release ``compute_teacher_encoder_observations`` mapped to Mini-Pi.

      [0:6)   foot contact forces (r, l; world xyz) / (m g)   release: wheel forces, N
      [6:7)   base height * 0.4 / LENGTH_RATIO                release: * 0.4
      [7:10)  root position minus env origin                  release: world pos
      [10:14) root quaternion (w, x, y, z), w >= 0             release: root quat
      [14:17) root linear velocity, world                     release: same
      [17:20) root angular velocity, world                    release: same

  Forces in body weights and height in stance units keep the release's numeric
  ranges on a 4x lighter, 2.2x shorter robot (ROBOT_ADAPTATION). The release's world
  position includes the Isaac Gym env-grid origin, which carries no information; the
  env-relative position is used (SIMULATOR_PORT).
  """
  robot = env.scene["robot"]
  data = robot.data
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces = sensor.data.force.reshape(env.num_envs, -1) / MINIPI_WEIGHT
  quat = data.root_link_quat_w
  quat = torch.where(quat[:, :1] < 0, -quat, quat)
  x = torch.cat(
    (
      forces,
      _base_height(env) * 0.4 / LENGTH_RATIO,
      data.root_link_pos_w - env.scene.env_origins,
      quat,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
    ),
    dim=-1,
  )
  return x * active_mask(env)


def ftsr_critic_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged critic state (50), release ``compute_privileged_observations``.

      [0:48)  noise-free o_t with the true base linear velocity * 2 in the zero slot
      [48:49) height sample clip(h - 0.5 LENGTH_RATIO, -1, 1) * 5
      [49:50) base height

  Release: 187 height-map samples ``clip(h_root - 0.5 - h_terrain, -1, 1) * 5`` on a
  17 x 11 grid. On the flat Mini-Pi ground every sample is the same function of the
  base height, so one sample is kept (SIMULATOR_PORT); the 0.5 m offset is scaled by
  the stance ratio (ROBOT_ADAPTATION).
  """
  o = _frame(env, true_lin_vel=True)
  h = _base_height(env)
  scan = torch.clamp(h - 0.5 * LENGTH_RATIO, -1.0, 1.0) * 5.0
  return torch.cat((o, scan, h), dim=-1) * active_mask(env)
