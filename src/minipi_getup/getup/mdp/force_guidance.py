"""Force-guided get-up terms, after FTSR (Hou et al., "Robust Fall Recovery for Armless
Bipedal-Wheeled Robots via Force-Guided Learning", RA-L 2026, arXiv:2606.14270).

- ``force_guided_assist``: an external upward force and uprighting torque on the torso
  whose size falls with torso height and with training progress (paper Eq. 4).
- ``assist_force_cost`` / ``assist_torque_cost``: the force and torque as costs. The
  paper folds them into the PPO advantage with fixed penalty factors (Eq. 5-8). With a
  fixed factor and GAE (linear in the reward) that equals a negative reward term.
  (The released code, github getup_gym, instead adds F_t - gamma * F_{t+1} to the TD
  error, potential-based shaping, and never fills the force buffer, so it is inactive.)
- ``height_stage``: height-progressive stages. The stage advances once 2/3 of the envs
  are above the current target height (or the batch mean passes 0.8 x target, as in
  the released code); each stage sets the target height (for the height reward and
  the assist) and a set of reward weights.
- ``height_banded_pose``: target pose by height band (released code's knee-down term).

The assist exists only in simulation: remove the event in play/eval configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch
from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# env.extras keys shared by the terms below.
_WRENCH_KEY = "assist_wrench"
_HEIGHT_CMD_KEY = "assist_height_cmd"


def _assist_wrench(env: ManagerBasedRlEnv) -> torch.Tensor:
  wrench = env.extras.get(_WRENCH_KEY)
  if wrench is None:
    return torch.zeros(env.num_envs, 6, device=env.device)
  return wrench


class force_guided_assist:
  """Height- and time-scaled upward force and uprighting torque on one body.

  k = k_h * clip(1 - (step - start_step) / (end_step - start_step), 0, 1), with
    k_h = 1 - exp(-mu * max(h_cmd - h, 0))   (``profile="exp"``, paper Eq. 4), or
    k_h = clip((h_cmd - h) / h_cmd, 0, 1)    (``profile="linear"``, the released code)
  F = k * f_max * z
  T = k * t_max * w, w = rotation vector taking the body z axis onto world z

  The paper's target rotation comes from target Euler angles; here it is the current
  orientation with the tilt removed, so the torque never acts on yaw (it is undefined
  when lying, and Mini-Pi already tends to spin). With ``follow_stage``, ``h_cmd``
  follows ``height_stage`` when that curriculum is present; the released code instead
  always scales by the final standing height.

  Use with ``mode="step"``: the wrench computed after a step acts during the next one.
  The wrench last applied is kept in ``env.extras["assist_wrench"]`` (N, 6) for the
  cost rewards and the critic observation.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._wrench = torch.zeros(env.num_envs, 6, device=env.device)
    env.extras[_WRENCH_KEY] = self._wrench

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._wrench[env_ids] = 0.0
    self._write()

  def _write(self) -> None:
    self._asset.write_external_wrench_to_sim(
      self._wrench[:, None, :3],
      self._wrench[:, None, 3:],
      body_ids=self._body_ids,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    f_max: float,
    t_max: float,
    mu: float,
    height_cmd: float,
    end_step: int,
    start_step: int = 0,
    profile: str = "exp",
    follow_stage: bool = True,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del env_ids  # Step mode: all envs.
    asset: Entity = env.scene[asset_cfg.name]
    h_cmd = env.extras.get(_HEIGHT_CMD_KEY, height_cmd) if follow_stage else height_cmd
    height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]
    progress = (env.common_step_counter - start_step) / (end_step - start_step)
    time_coeff = min(1.0, max(0.0, 1.0 - progress))
    if profile == "linear":
      k_h = torch.clamp((h_cmd - height) / h_cmd, 0.0, 1.0)
    else:
      k_h = 1.0 - torch.exp(-mu * torch.clamp(h_cmd - height, min=0.0))
    k = k_h * time_coeff

    # Rotation vector (world frame) of the shortest rotation taking body z onto world z.
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    w, x, y, _ = quat.unbind(-1)
    z = quat[:, 3]
    body_z = torch.stack(
      (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), dim=-1
    )
    world_z = torch.zeros_like(body_z)
    world_z[:, 2] = 1.0
    axis = torch.cross(body_z, world_z, dim=-1)
    sin = axis.norm(dim=-1, keepdim=True)
    cos = body_z[:, 2:3]
    angle = torch.atan2(sin, cos)
    rotvec = axis / sin.clamp(min=1e-6) * angle

    self._wrench[:, 2] = k * f_max
    self._wrench[:, 3:] = (k * t_max).unsqueeze(-1) * rotvec
    self._write()

    log = env.extras.setdefault("log", {})
    log["Assist/force_mean"] = self._wrench[:, 2].mean().item()
    log["Assist/time_coeff"] = time_coeff


def assist_force_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Magnitude of the assist force (N) applied during the last step."""
  return _assist_wrench(env)[:, :3].norm(dim=-1)


def assist_torque_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Magnitude of the assist torque (Nm) applied during the last step."""
  return _assist_wrench(env)[:, 3:].norm(dim=-1)


def assist_wrench_obs(env: ManagerBasedRlEnv, scale: float = 0.05) -> torch.Tensor:
  """Assist wrench (privileged, critic only), scaled to roughly unit size."""
  return _assist_wrench(env) * scale


def body_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """World height of a body (privileged, critic only)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2:3]


def leg_mirror_error(
  env: ManagerBasedRlEnv,
  same_sign: tuple[str, ...],
  mirrored: tuple[str, ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Paper's "leg bias": L2 norm of the left/right difference of mirrored joints.

  All Mini-Pi joint axes point the same way on both legs, so pitch-type joints mirror
  as q_l = q_r and roll/yaw-type joints as q_l = -q_r. Names are suffixes after the
  ``l_``/``r_`` prefix.
  """
  asset: Entity = env.scene[asset_cfg.name]
  names = asset.joint_names
  q = asset.data.joint_pos
  diffs = []
  for suffix in same_sign:
    diffs.append(q[:, names.index("l_" + suffix)] - q[:, names.index("r_" + suffix)])
  for suffix in mirrored:
    diffs.append(q[:, names.index("l_" + suffix)] + q[:, names.index("r_" + suffix)])
  return torch.stack(diffs, dim=-1).norm(dim=-1)


def low_for_too_long(
  env: ManagerBasedRlEnv,
  max_time_s: float,
  min_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Paper's termination: still on the ground after ``max_time_s``."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]
  late = env.episode_length_buf * env.step_dt > max_time_s
  return late & (height < min_height)


class height_banded_pose:
  """Reward a target joint pose chosen by torso height band (the released code's
  ``rew_knee_down``: a deep crouch while low, the nominal pose higher up).

  ``bands`` is a list of ``(min_height, max_height, {joint regex: angle})``. Outside
  every band the reward is 0. Reward = exp(-mean |q - q_target| / std).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    _, names = asset.find_joints(asset_cfg.joint_names)
    self._bounds = []
    targets = []
    for low, high, pose in cfg.params["bands"]:
      _, _, values = resolve_matching_names_values(pose, names)
      self._bounds.append((low, high))
      targets.append(torch.tensor(values, device=env.device))
    self._targets = targets

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    bands: list,
    std: float,
    asset_cfg: SceneEntityCfg,
    height_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del bands
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.body_link_pos_w[:, height_cfg.body_ids[0], 2]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    reward = torch.zeros_like(height)
    for (low, high), target in zip(self._bounds, self._targets, strict=True):
      err = (q - target).abs().mean(dim=-1)
      in_band = (height >= low) & (height < high)
      reward = torch.where(in_band, torch.exp(-err / std), reward)
    return reward


class HeightStage(TypedDict):
  height_cmd: float
  weights: dict[str, float]


class height_stage:
  """Height-progressive stages (paper Sec. II-D).

  Stage i has a target height ``height_cmd``. With ``rule="fraction"`` (paper) the next
  stage starts once more than ``fraction`` of the envs are above it; with
  ``rule="mean"`` (released code) once the batch mean height reaches
  ``mean_factor * height_cmd``. Checked on every reset. The new stage's reward weights
  are applied, and its target height goes to the assist (``env.extras``) and to the
  ``height_reward`` named by ``height_reward_name``. Stages only advance.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._stages: list[HeightStage] = cfg.params["stages"]
    self._stage = -1
    self._apply(env, 0, cfg.params["height_reward_name"])

  def _apply(self, env: ManagerBasedRlEnv, stage: int, height_reward_name: str) -> None:
    self._stage = stage
    spec = self._stages[stage]
    env.extras[_HEIGHT_CMD_KEY] = spec["height_cmd"]
    height_cfg = env.reward_manager.get_term_cfg(height_reward_name)
    height_cfg.params["desired_height"] = spec["height_cmd"]
    for name, weight in spec["weights"].items():
      env.reward_manager.get_term_cfg(name).weight = weight

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[HeightStage],
    height_reward_name: str,
    fraction: float = 2.0 / 3.0,
    rule: str = "fraction",
    mean_factor: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> dict[str, torch.Tensor]:
    del env_ids
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], 2]
    h_cmd = stages[self._stage]["height_cmd"]
    above = (height > h_cmd).float().mean().item()
    if rule == "mean":
      advance = height.mean().item() >= mean_factor * h_cmd
    else:
      advance = above > fraction
    if advance and self._stage + 1 < len(stages):
      self._apply(env, self._stage + 1, height_reward_name)
    return {
      "stage": torch.tensor(float(self._stage)),
      "frac_above_target": torch.tensor(above),
    }
