"""CPU geometry/logic checks; --gpu adds physical sensor, replay and two PPO updates."""

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .geometry import XML, calibrate
from .rewards import StageLogic, orientation


def feature_pose(name="upright", device="cpu", n=1):
  geo = calibrate()
  m = mujoco.MjModel.from_xml_path(str(XML))
  d = mujoco.MjData(m)
  axis, angle = {
    "upright": ([0, 1, 0], 0),
    "supine": ([0, 1, 0], -math.pi / 2),
    "prone": ([0, 1, 0], math.pi / 2),
    "left": ([1, 0, 0], math.pi / 2),
    "right": ([1, 0, 0], -math.pi / 2),
    "situp": ([0, 1, 0], -math.pi / 4),
  }[name]
  d.qpos[:] = 0
  d.qpos[2] = geo["standing_height"]
  mujoco.mju_axisAngle2Quat(d.qpos[3:7], np.array(axis, dtype=float), angle)
  mujoco.mj_forward(m, d)
  gravity = d.xmat[1].reshape(3, 3).T @ np.array([0, 0, -1.0])

  def t(x):
    return (
      torch.tensor(x, device=device, dtype=torch.float).expand(n, *np.shape(x)).clone()
    )

  return dict(
    gravity=t(gravity),
    height=t(geo["standing_height"]),
    head_height=t(geo["standing_height"] + 0.08),
    foot_load=t([0.5, 0.5]),
    foot_up=t([1.0, 1.0]),
    placement=t(1.0),
    balance_error=t(0.0),
    support_pose=t(1.0),
    nominal_pose=t(1.0),
    pose_max=t(0.0),
    symmetry_rms=t(0.0),
    overshoot=t(0.0),
    ang_speed=t(0.0),
    lin_speed=t(0.0),
    offaxis_speed=t(0.0),
  )


def cpu_checks():
  geo = calibrate()
  assert len(geo["support_poses"]) >= 3
  print(
    "GEOMETRY",
    geo["standing_height"],
    geo["support_heights"],
    "IK errors",
    geo["ik_error_m"],
  )
  expected = {
    "upright": [0, 0, -1],
    "supine": [-1, 0, 0],
    "prone": [1, 0, 0],
    "left": [0, -1, 0],
    "right": [0, 1, 0],
  }
  for name, g in expected.items():
    f = feature_pose(name)
    assert torch.allclose(f["gravity"], torch.tensor([g], dtype=torch.float), atol=1e-6)
    print(
      "ORIENTATION",
      name,
      f["gravity"].tolist(),
      "up/side/prone",
      tuple(v.tolist() for v in orientation(f["gravity"])),
    )
  active = torch.ones(1, dtype=torch.bool)
  # Pure sagittal erection has zero lateral inclination at every pitch.
  for pitch in np.linspace(-math.pi / 2, 0, 25):
    gravity = torch.tensor([[math.sin(pitch), 0.0, -math.cos(pitch)]])
    assert orientation(gravity)[1].item() == 0
  scores = {}
  for name in ("supine", "prone", "left", "right", "situp", "upright"):
    logic = StageLogic(1, "cpu", 0.02, geo)
    f = feature_pose(name)
    if name != "upright":
      f["foot_load"] *= 0
    for _ in range(80):
      r = logic.update(f, active)
    scores[name] = (r["style"].item(), r["target"].item())
  assert scores["prone"][0] < scores["supine"][0]
  assert scores["left"][0] < scores["situp"][0]
  assert scores["upright"][1] > 3.9
  print("STATIONARY_REWARDS style/target per-second", scores)
  # Kinematically validated support references, with idealized balanced loads.
  # These test reward ranking, not dynamic reachability or motor sufficiency.
  for height, q in zip(geo["support_heights"], geo["support_poses"], strict=True):
    logic = StageLogic(1, "cpu", 0.02, geo)
    f = feature_pose()
    q = torch.tensor(q, dtype=torch.float)
    f["height"][:] = height
    f["head_height"][:] = height + 0.08
    f["nominal_pose"][:] = torch.exp(-q.square().mean() / 0.3**2)
    f["pose_max"][:] = q.abs().max()
    for _ in range(80):
      r = logic.update(f, active)
    assert r["progress"].item() == 0
    if height < 0.9 * geo["standing_height"]:
      assert logic.stage.item() == 1 and not r["success"].any()
      assert r["target"].item() < scores["upright"][1] / 10
    print(
      "IK_POSE_REWARD",
      height,
      "stage",
      logic.stage.item() + 1,
      "target",
      r["target"].item(),
    )
  logic = StageLogic(2, "cpu", 0.02, geo)
  f = feature_pose(n=2)
  f["gravity"][1] = torch.tensor([-1.0, 0.0, 0.0])
  f["height"][1] = 0.08
  f["head_height"][1] = 0.08
  f["foot_load"][1] = 0
  rows = []
  for _ in range(80):
    r = logic.update(f, torch.ones(2, dtype=torch.bool))
    rows.append([*logic.stage.tolist(), *r["style"].tolist(), *r["target"].tolist()])
  assert logic.stage.tolist() == [2, 0] and logic.ever_stable.tolist() == [True, False]
  rows = np.array(rows)
  assert np.max(np.abs(np.diff(rows[:, 2:], axis=0))) < 0.1
  # No reward discontinuity specifically when stages change at a held pose.
  logic.reset(torch.tensor([0]))
  assert logic.stage.tolist() == [0, 0] and logic.hold.sum() == 0
  logic = StageLogic(1, "cpu", 0.02, geo)
  f = feature_pose()
  for _ in range(80):
    logic.update(f, active)
  f["foot_load"] *= 0
  for _ in range(20):
    r = logic.update(f, active)
  assert logic.stage.item() == 2 and logic.hold.item() == 0 and r["target"].item() == 0
  f["foot_load"][:] = 0.5
  for _ in range(80):
    r = logic.update(f, active)
  assert (
    logic.hold.item() > 1
    and r["progress"].item() == 0
    and logic.transitions.item() == 2
  )
  # Repeated posture excursions cannot regenerate high-water rewards.
  for _ in range(3):
    logic.update(feature_pose("supine"), active)
    r = logic.update(feature_pose(), active)
    assert r["progress"].item() == 0
  f = feature_pose()
  f["overshoot"][:] = 0.025
  for _ in range(80):
    r = logic.update(f, active)
  assert not r["success"].any() and r["target"].item() < 0.3
  print(
    "PASS CPU: pose signs, sagittal motion, route costs, per-env dwell, smooth rewards, reset, loss/recovery, no progress farming, invalid-limit stance"
  )


def gpu_checks(checkpoint=None, trace=None):
  from warp._src.build import init_kernel_cache

  init_kernel_cache("/tmp/host-supine-narrow-warp")
  import minipi_getup.host.env as baseline
  from minipi_getup.host.clpai_asset import use_clpai_asset
  from minipi_getup.host.config import PiCfg, class_to_dict
  from minipi_getup.host.evaluate import load_policy
  from minipi_getup.host.train_clpai import REAL_KD, REAL_KP, REAL_TORQUE
  from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner

  from .config import StageCfg, StagePPOCfg
  from .env import StageEnv

  torch.set_num_threads(4)
  torch.manual_seed(1)
  c = StageCfg()
  e = StageEnv(c, 64)
  assert baseline.TORQUE_LIMITS is None
  assert e.num_obs == 258 and e.rew_buf.shape == (64, 4)
  assert e.dt == 0.02 and e.sim.mj_model.opt.timestep == 0.0025
  assert e.cfg.rewards.reward_group_weights == [2.5, 0.1, 1, 1]
  assert torch.allclose(
    e.torque_limits,
    torch.tensor([16, 16, 16, 16, 16, 6] * 2, device=e.device, dtype=torch.float),
  )
  # Compile the actual baseline-real plant and compare all physical arrays.
  saved = (baseline.ASSET_XML, baseline.get_robot_cfg, baseline.TORQUE_LIMITS)
  use_clpai_asset()
  baseline.TORQUE_LIMITS = dict(REAL_TORQUE)
  bc = PiCfg()
  bc.control.stiffness = dict(REAL_KP)
  bc.control.damping = dict(REAL_KD)
  be = baseline.LeggedRobot_Pi(bc, 1, "cuda:0")
  baseline.ASSET_XML, baseline.get_robot_cfg, baseline.TORQUE_LIMITS = saved
  for field in (
    "body_mass",
    "body_inertia",
    "body_pos",
    "body_quat",
    "jnt_range",
    "jnt_axis",
    "geom_type",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_contype",
    "geom_conaffinity",
    "dof_armature",
    "actuator_forcerange",
    "actuator_trnid",
  ):
    assert np.array_equal(
      getattr(e.sim.mj_model, field), getattr(be.sim.mj_model, field)
    ), field
  print("PASS physical model parity, four critics, unchanged observation/rates/plant")
  # Write standard poses to the REAL GPU simulation and read projected gravity.
  for name in ("supine", "prone", "left", "right", "upright"):
    f = feature_pose(name)
    q = torch.zeros(64, 4, device=e.device)
    angle = {
      "supine": -math.pi / 2,
      "prone": math.pi / 2,
      "left": math.pi / 2,
      "right": -math.pi / 2,
      "upright": 0,
    }[name]
    q[:, 0] = math.cos(angle / 2)
    q[:, 1 if name in ("left", "right") else 2] = math.sin(angle / 2)
    root = e.root_states.clone()
    root[:, :3] = 0
    root[:, 2] = e.geometry["standing_height"] - 0.002
    root[:, 3:7] = q
    root[:, 7:] = 0
    e.robot.write_root_state_to_sim(root)
    e.robot.write_joint_state_to_sim(
      torch.zeros_like(e.dof_pos), torch.zeros_like(e.dof_vel)
    )
    e.sim.forward()
    e._refresh_root_and_body_states()
    from mjlab.utils.lab_api.math import quat_apply_inverse

    measured = quat_apply_inverse(e.root_states[:, 3:7], e.gravity_vec)
    assert torch.allclose(
      measured, f["gravity"].to(e.device).expand_as(measured), atol=1e-5
    )
    if name == "upright":
      e.base_pos[:] = e.root_states[:, :3]
      e.base_quat[:] = e.root_states[:, 3:7]
      e.projected_gravity[:] = measured
      fs = e.measure()
      print("NATIVE_FOOT_LOAD", fs["foot_load"][0].tolist())
      assert (fs["foot_load"] > 0).all()
  print("PASS native GPU pose signs and foot-ground force sensor")
  # Two actual PPO updates with original network, minibatches, epochs and DR.
  pc = StagePPOCfg()
  r = OnPolicyRunner(e, c, class_to_dict(pc), None, device="cuda:0")
  # Force timeout during smoke to exercise vectorized reset/stage cleanup.
  e.episode_length_buf[:] = e.max_episode_length - 10
  for update in range(2):
    with torch.inference_mode():
      obs = e.get_observations()
      for _ in range(r.num_steps_per_env):
        actions = r.alg.act(obs, obs)
        obs, _, reward, done, info = e.step(actions)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        r.alg.process_env_step(reward, done, info)
      r.alg.compute_returns(obs)
    assert torch.isfinite(r.alg.storage.advantages).all()
    loss = r.alg.update()
    assert np.isfinite(loss).all()
    assert all(
      p.grad is None or torch.isfinite(p.grad).all()
      for p in r.alg.actor_critic.parameters()
    )
    print(
      "PPO_UPDATE",
      update,
      "loss",
      loss,
      "reward_shape",
      tuple(reward.shape),
      "actor_obs",
      tuple(obs.shape),
    )
  print(
    "PASS finite rewards/returns/advantages/gradients, actual PPO tensor dimensions, timeout resets"
  )
  if checkpoint:
    # Fresh process recommended for standalone evaluation; here config class
    # instances are separate, so evaluation only disables DR in this instance.
    ec = StageCfg()
    ec.noise.add_noise = False
    ec.curriculum.pull_force = False
    ec.curriculum.force = 0
    for k in dir(ec.domain_rand):
      if k.startswith("randomize_") or k == "delay":
        setattr(ec.domain_rand, k, False)
    ss = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ec.control.action_scale = float(
      ss["infos"]["curriculum_state"]["action_rescale"].mean()
    )
    ev = StageEnv(ec, 1)
    policy = load_policy(str(checkpoint), ev)
    rows = []
    with torch.no_grad():
      obs, _ = ev.reset()
      for _j in range(490):
        obs, _, reward, done, _ = ev.step(policy.act_inference(obs))
        if done.any():
          break
        f = ev.features
        logic = ev.stage_logic
        rows.append(
          [
            float(ev.real_episode_length_buf[0] * ev.dt),
            float(f["height"][0]),
            float(f["gravity"][0, 0]),
            float(f["gravity"][0, 1]),
            float(logic.last["up"][0]),
            float(logic.last["side"][0] * 180 / math.pi),
            int(logic.last["prone"][0]),
            int(logic.stage[0]) + 1,
            float(f["foot_load"][0, 0]),
            float(f["foot_load"][0, 1]),
            float(f["overshoot"][0]),
            int(logic.last["stable"][0]),
            *reward[0].cpu().tolist(),
          ]
        )
    a = np.array(rows)
    act = a[:, 0] > 0.62
    for label, mask in [
      ("side>30", a[:, 5] > 30),
      ("side>60", a[:, 5] > 60),
      ("prone", a[:, 6] > 0),
      ("upright", (a[:, 4] > 0.8) & (a[:, 1] > 0.3173)),
    ]:
      hit = a[mask & act]
      print(
        "BASELINE_REPLAY",
        label,
        "first_s",
        hit[0, 0] if len(hit) else None,
        "fraction",
        float((mask & act).sum() / act.sum()),
      )
    print(
      "BASELINE_REPLAY final",
      a[-1].tolist(),
      "ever_stable",
      ev.stage_logic.ever_stable.item(),
      "peak_tau",
      ev.peak_torque.max().item(),
      "peak_qdot",
      ev.peak_joint_vel.max().item(),
      "max_overshoot",
      ev.max_overshoot.item(),
    )
    if trace:
      import csv

      with open(trace, "w") as out:
        w = csv.writer(out)
        w.writerow(
          [
            "t",
            "base_z",
            "gravity_x",
            "gravity_y",
            "up",
            "side_deg",
            "prone",
            "stage",
            "left_load_bw",
            "right_load_bw",
            "overshoot",
            "stable",
            "task",
            "regu",
            "style",
            "target",
          ]
        )
        w.writerows(rows)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--gpu", action="store_true")
  ap.add_argument("--checkpoint", type=Path)
  ap.add_argument("--trace", type=Path)
  a = ap.parse_args()
  cpu_checks()
  if a.gpu:
    gpu_checks(a.checkpoint, a.trace)


if __name__ == "__main__":
  main()
