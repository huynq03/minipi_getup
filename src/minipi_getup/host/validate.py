"""Pre-training checks for the HoST port (timing, actions, observations, force, rewards,
multi-critic PPO shapes, reset pose). Prints a report and writes reset-pose renders.

python -m minipi_getup.host.validate [--out results/host/validation]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from minipi_getup.host.config import PiCfg, PiCfgPPO, class_to_dict

DEV = "cuda:0"
RESULTS: dict = {}


def check(name: str, ok: bool, detail) -> None:
  RESULTS[name] = {"ok": bool(ok), "detail": detail}
  print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def no_dr(cfg: PiCfg) -> PiCfg:
  dr = cfg.domain_rand
  for k in dir(dr):
    if k.startswith("randomize_") or k == "delay":
      setattr(dr, k, False)
  cfg.noise.add_noise = False
  return cfg


def standing_pose(env, ids):
  """Zero joints, upright, feet on the ground (base 0.351 m as in init_state)."""
  rs = torch.zeros(len(ids), 13, device=DEV)
  rs[:, 2] = 0.351
  rs[:, 3] = 1.0
  env.robot.write_root_state_to_sim(rs, env_ids=ids)
  z = torch.zeros(len(ids), env.num_dof, device=DEV)
  env.robot.write_joint_state_to_sim(z, z, env_ids=ids)
  env.dof_pos[ids] = 0
  env.dof_vel[ids] = 0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="results/host/validation")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  torch.manual_seed(0)
  from minipi_getup.host.env import STAND_HEIGHT, LeggedRobot_Pi

  # ---------------------------------------------------------------- timing
  env = LeggedRobot_Pi(no_dr(PiCfg()), 4, DEV)
  check(
    "timing",
    env.cfg.sim.dt == 0.005
    and abs(env.sim.mj_model.opt.timestep * env.physics_substeps - 0.005) < 1e-12
    and env.cfg.control.decimation == 4
    and abs(env.dt - 0.02) < 1e-12
    and env.max_episode_length == 500
    and env.unactuated_time == 30,
    {
      "control_substep_dt (Isaac sim.dt)": env.cfg.sim.dt,
      "mujoco_timestep": float(env.sim.mj_model.opt.timestep),
      "mujoco_steps_per_control_substep": env.physics_substeps,
      "decimation": env.cfg.control.decimation,
      "policy_dt": env.dt,
      "max_episode_length": float(env.max_episode_length),
      "episode_steps_incl_timeout_step": int(env.max_episode_length) + 1,
      "unactuated_steps": env.unactuated_time,
      "unactuated_s": env.unactuated_time * env.dt,
    },
  )

  # ---------------------------------------------------------------- reset + unactuated
  env.reset()  # reset_idx(all) + one zero-action step (BaseTask.reset)
  a = torch.full((4, 12), 0.7, device=DEV)
  zero_obs, acts_zero = [], []
  for _ in range(29):
    obs, *_ = env.step(a)
    zero_obs.append(float(obs.abs().max()))
    acts_zero.append(float(env.actions.abs().max()))
  check(
    "unactuated_period",
    max(zero_obs) == 0
    and max(acts_zero) == 0
    and int(env.real_episode_length_buf[0]) == 30,
    {
      "steps_checked": 29,
      "max_|obs|": max(zero_obs),
      "max_|action|": max(acts_zero),
      "real_episode_length_after": int(env.real_episode_length_buf[0]),
    },
  )
  # Release order: step() masks with real_episode_length_buf *before* the increment,
  # observations after it. So the step that ends at 31 already observes (obs != 0) but
  # still sends zero actions; the next step is the first actuated one.
  obs, *_ = env.step(a)
  obs_on = float(obs[:, -43:].abs().max()) > 0
  act_still_zero = float(env.actions.abs().max()) == 0
  env.step(a)
  act_on = float(env.actions.abs().max()) > 0
  check(
    "first_observed_and_actuated_step",
    obs_on and act_still_zero and act_on,
    {
      "first_nonzero_obs_after_step": 31,
      "first_nonzero_action_in_step": 32,
      "first_actuation_time_s": 31 * env.dt,
    },
  )

  # ---------------------------------------------------------------- action semantics
  res = {}
  for scale in (1.0, 0.25):
    env.action_rescale[:] = scale
    q_before = env.dof_pos.clone()
    acts = torch.full((4, 12), 0.1, device=DEV)
    env.actions = acts.clone()
    tq = env._compute_torques(env.actions)
    diff = env.joint_pos_target - env.dof_pos
    expect_tq = torch.clip(
      env.p_gains * 0.1 * scale - env.d_gains * env.dof_vel, -20, 20
    )
    res[scale] = {
      "q_target-q": [float(diff.min()), float(diff.max())],
      "max|torque-expected|": float((tq - expect_tq).abs().max()),
      "q_unchanged": bool(torch.equal(q_before, env.dof_pos)),
    }
  ok = all(
    abs(res[s]["q_target-q"][0] - 0.1 * s) < 1e-6
    and abs(res[s]["q_target-q"][1] - 0.1 * s) < 1e-6
    and res[s]["max|torque-expected|"] < 1e-4
    for s in res
  )
  check("action_semantics", ok, {str(k): v for k, v in res.items()})
  env.action_rescale[:] = 1.0

  # delay: buffer index i holds the scaled action pushed (4 - i) substeps ago
  env2 = LeggedRobot_Pi(no_dr(PiCfg()), 5, DEV)
  env2.cfg.domain_rand.delay = True
  env2.delay_idx = torch.arange(5, device=DEV)
  env2.delay_buffer.zero_()
  outs = []
  for k in range(1, 6):
    env2._compute_torques(torch.full((5, 12), float(k), device=DEV))
    outs.append((env2.joint_pos_target - env2.dof_pos)[:, 0].tolist())
  # after pushing 1..5, delay_idx 4 -> 5 (no delay), idx 0 -> 1 (4 substeps old)
  last = [round(v) for v in outs[-1]]
  check(
    "delay_buffer",
    last == [1, 2, 3, 4, 5],
    {
      "target_offsets_after_5_pushes_by_delay_idx_0..4": last,
      "delay_substeps": [4, 3, 2, 1, 0],
    },
  )
  del env2

  # ---------------------------------------------------------------- observations
  env.reset_idx(torch.arange(4, device=DEV))
  for _ in range(40):
    obs, *_ = env.step(torch.zeros(4, 12, device=DEV))
  frame = obs[0, -43:]
  ref = torch.cat(
    (
      env.base_ang_vel[0] * 0.25,
      env.projected_gravity[0],
      env.dof_pos[0],
      env.dof_vel[0] * 0.05,
      env.actions[0],
    )
  )
  hist_shift = (
    torch.equal(obs[:, : 43 * 5], env._prev_obs_for_check[:, 43:])
    if hasattr(env, "_prev_obs_for_check")
    else None
  )
  prev = obs.clone()
  obs2, *_ = env.step(torch.zeros(4, 12, device=DEV))
  shift_ok = torch.equal(obs2[:, : 43 * 5], prev[:, 43:])
  check(
    "observation_layout",
    obs.shape == (4, 258)
    and env.num_one_step_obs == 43
    and env.actor_history_length == 6
    and float((frame[:42] - ref).abs().max()) < 1e-5
    and abs(float(frame[42]) - 1.0) <= 0.025
    and shift_ok,
    {
      "shape": list(obs.shape),
      "frame": 43,
      "history": 6,
      "order": "ang_vel*0.25(3) | proj_gravity(3) | dof_pos(12) | dof_vel*0.05(12) | actions(12) | action_rescale+U(+-0.025)(1)",
      "max|frame-ref|(no noise)": float((frame[:42] - ref).abs().max()),
      "action_rescale_obs": float(frame[42]),
      "history_shift_ok": bool(shift_ok),
      "_unused": hist_shift,
    },
  )
  nv = env.noise_scale_vec.tolist()
  check(
    "observation_noise_vector",
    np.allclose(nv[0:3], 0.05)
    and np.allclose(nv[3:6], 0.05)
    and np.allclose(nv[6:18], 0.01)
    and np.allclose(nv[18:30], 0.075)
    and np.allclose(nv[30:43], 0.0),
    {
      "ang_vel": nv[0],
      "gravity": nv[3],
      "dof_pos": nv[6],
      "dof_vel": nv[18],
      "actions+rescale": nv[30],
    },
  )

  # ---------------------------------------------------------------- force
  envf = LeggedRobot_Pi(no_dr(PiCfg()), 3, DEV)
  envf.reset()
  ids = torch.arange(3, device=DEV)
  for _ in range(35):  # past the unactuated window, robot lying (pg_z ~ 0)
    envf.step(torch.zeros(3, 12, device=DEV))
  lying_force = envf.robot.data.body_external_force[
    :, 0
  ].clone()  # applied in the last substep
  lying_pending = envf.pending_force[:, 0].clone()
  # make env 0 and 1 upright in the air (no contacts), env 2 lying; env 1 has force 0
  rs = torch.zeros(3, 13, device=DEV)
  rs[:, 2] = 2.0
  rs[:, 3] = 1.0
  envf.robot.write_root_state_to_sim(rs[:2], env_ids=ids[:2])
  envf.force[1] = 0.0
  envf.sim.forward()
  envf.post_physics_step_refresh_only = True
  envf._refresh_root_and_body_states()
  envf.base_quat[:] = envf.root_states[:, 3:7]
  from mjlab.utils.lab_api.math import quat_apply_inverse

  envf.projected_gravity[:] = quat_apply_inverse(envf.base_quat, envf.gravity_vec)
  m = envf.sim.model.body_mass[0]  # per-world masses (world 0 = env 0)
  M = float(envf.sim.model.body_mass[0, envf.robot.indexing.body_ids.long()].sum())

  def com_vz(e):
    bid = e.robot.indexing.body_ids.long()
    mass = e.sim.model.body_mass[:, bid]
    vz = e.robot.data.body_com_lin_vel_w[:, :, 2]
    return (mass * vz).sum(-1) / mass.sum(-1)

  envf.pending_force.zero_()
  envf.actions = torch.zeros(3, 12, device=DEV)
  # one substep to compute the pending force from the upright gate, then measure
  envf.real_episode_length_buf[:] = 100
  envf.robot.data.write_external_wrench(
    envf.pending_force, None, body_ids=envf.base_indices
  )
  # emulate step(): substep 1 computes pending force after simulate
  v0 = None
  accs = []
  forces_seen = []
  for _ in range(4):
    envf.torques = envf._compute_torques(envf.actions)
    envf.robot.set_joint_effort_target(envf.torques * 0)
    envf.robot.write_data_to_sim()
    envf.robot.data.write_external_wrench(
      envf.pending_force, None, body_ids=envf.base_indices
    )
    forces_seen.append(envf.robot.data.body_external_force[:, 0, 2].tolist())
    envf.sim.forward()
    v0 = com_vz(envf)
    for _ in range(envf.physics_substeps):
      envf.sim.step()
    envf.sim.forward()
    v1 = com_vz(envf)
    accs.append(((v1 - v0) / 0.005).tolist())
    envf._refresh_dof_state_tensor()
    f = torch.zeros(3, 1, 3, device=DEV)
    f[:, :, 2] = envf.force
    f *= (envf.real_episode_length_buf.unsqueeze(1) > envf.unactuated_time).unsqueeze(1)
    f *= (envf.projected_gravity[:, 2] < -0.8).unsqueeze(1).unsqueeze(1)
    envf.pending_force = f
  a_last = accs[-1]
  expected = 15.0 / M
  check(
    "pull_force",
    float(lying_pending.abs().max()) == 0
    and forces_seen[0] == [0.0, 0.0, 0.0]
    and abs(forces_seen[1][0] - 15.0) < 1e-6
    and forces_seen[1][1] == 0.0
    and forces_seen[1][2] == 0.0
    and abs((a_last[0] - a_last[1]) - expected) < 0.02 * expected,
    {
      "lying_after_unactuated_force(N)": lying_pending[:, 2].tolist(),
      "xfrc_z per substep [env0 upright F=15, env1 upright F=0, env2 lying]": forces_seen,
      "com_az env0-env1 (m/s^2)": a_last[0] - a_last[1],
      "expected 15/M": expected,
      "robot_mass": M,
      "body": envf.body_names[envf.base_indices[0]],
      "frame": "world +Z at body COM (xfrc_applied)",
      "_": float(m.sum()),
    },
  )
  _ = lying_force
  del envf

  # ---------------------------------------------------------------- rewards
  envr = LeggedRobot_Pi(no_dr(PiCfg()), 2, DEV)
  envr.reset()
  for _ in range(40):
    envr.step(torch.zeros(2, 12, device=DEV))
  lying = {k: float(v[0]) for k, v in envr.term_values.items()}
  lying_groups = envr.rew_buf[0].tolist()
  standing_pose(envr, torch.tensor([1], device=DEV))
  envr.sim.forward()
  envr.step(torch.zeros(2, 12, device=DEV))
  st = {k: float(v[1]) for k, v in envr.term_values.items()}
  st_groups = envr.rew_buf[1].tolist()
  finite = all(
    np.isfinite(list(lying.values()) + list(st.values()) + lying_groups + st_groups)
  )
  check(
    "rewards",
    finite and st["task_orientation"] > 0.9 and lying["task_orientation"] < 0.1,
    {
      "groups": envr.reward_groups,
      "lying_supine_groups": lying_groups,
      "standing_groups": st_groups,
      "standing_head_minus_feet": float(envr.old_headheight[1]),
      "standing_base_z": float(envr.root_states[1, 2]),
      "stand_height_threshold": STAND_HEIGHT,
      "lying_terms": lying,
      "standing_terms": st,
    },
  )
  del envr

  # ---------------------------------------------------------------- PPO shapes
  from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner

  envp = LeggedRobot_Pi(PiCfg(), 64, DEV)
  tcfg = class_to_dict(PiCfgPPO())
  runner = OnPolicyRunner(envp, PiCfg(), tcfg, log_dir=None, device=DEV)
  ac = runner.alg.actor_critic
  arch = str(ac)
  obs = envp.get_observations()
  with torch.inference_mode():
    for _ in range(tcfg["runner"]["num_steps_per_env"]):
      act = runner.alg.act(obs, obs)
      obs, _, rew, dones, infos = envp.step(act)
      runner.alg.process_env_step(rew, dones, infos)
    runner.alg.compute_returns(obs)
  st_ = runner.alg.storage
  adv = st_.advantages.flatten(0, 1)
  check(
    "multi_critic_ppo",
    tuple(st_.rewards.shape[1:]) == (64, 4)
    and tuple(st_.values.shape[1:]) == (64, 4)
    and tuple(st_.advantages.shape[1:]) == (64, 4)
    and tuple(st_.multi_critic_advantages.shape[1:]) == (64,)
    and torch.allclose(adv.mean(0), torch.zeros(4, device=DEV), atol=1e-4)
    and torch.allclose(adv.std(0), torch.ones(4, device=DEV), atol=1e-3)
    and torch.allclose(
      st_.multi_critic_advantages,
      (st_.advantages * torch.tensor([2.5, 0.1, 1, 1], device=DEV)).sum(-1),
    ),
    {
      "rewards": list(st_.rewards.shape),
      "values": list(st_.values.shape),
      "advantages": list(st_.advantages.shape),
      "combined_advantage": list(st_.multi_critic_advantages.shape),
      "adv_mean_per_critic": adv.mean(0).tolist(),
      "adv_std_per_critic": adv.std(0).tolist(),
      "storage_step_filled": st_.step,
      "init_std": ac.std.tolist()[:3],
      "num_critics": ac.num_critics,
      "smoothness_coefs": {
        "policy": runner.alg.smoothness_upper_bound
        * runner.alg.smoothness_lower_bound
        / (runner.alg.smoothness_upper_bound - runner.alg.smoothness_lower_bound),
      },
    },
  )
  with open(os.path.join(args.out, "architecture.txt"), "w") as f:
    f.write(arch)
  print(arch)
  try:
    runner.alg.update()
    check("ppo_update_runs", True, runner.alg.stats)
  except Exception as e:  # noqa: BLE001
    check("ppo_update_runs", False, repr(e))
  del runner, envp

  # ---------------------------------------------------------------- reset pose renders
  from minipi_getup.host.render import Renderer, write_video

  envv = LeggedRobot_Pi(PiCfg(), 6, DEV)  # full training DR
  envv.reset_idx(torch.arange(6, device=DEV))
  envv.sim.forward()
  rend = Renderer(envv.sim.mj_model)
  qs = [envv.sim.data.qpos.clone().cpu().numpy()]
  pg = []
  for _ in range(40):
    envv.step(torch.zeros(6, 12, device=DEV))
    qs.append(envv.sim.data.qpos.clone().cpu().numpy())
    pg.append(envv.projected_gravity.clone().cpu().numpy())
  import imageio.v2 as imageio

  for k in range(6):
    for t, tag in ((0, "reset"), (30, "settled_0.6s")):
      img = rend.frame(qs[t][k])
      img = Renderer.overlay(img, [f"env {k} {tag}", f"base z {qs[t][k][2]:.3f}"])
      imageio.imwrite(os.path.join(args.out, f"reset_env{k}_{tag}.png"), img)
  write_video(
    os.path.join(args.out, "reset_drop_env0.mp4"),
    [rend.frame(q[0]) for q in qs],
    fps=50,
  )
  settled = pg[29]
  from mjlab.utils.lab_api.math import quat_apply_inverse

  q0 = envv.base_init_state[3:7].unsqueeze(0)
  pg0 = quat_apply_inverse(q0, torch.tensor([[0.0, 0.0, -1.0]], device=DEV))[0]
  check(
    "reset_pose_supine",
    bool(pg0[0] < -0.999),
    {
      "init_quat_wxyz": envv.base_init_state[3:7].tolist(),
      "projected_gravity_at_reset": pg0.tolist(),
      "interpretation": "gravity along -x_body (out of the back): supine, face up",
      "projected_gravity_after_0.6s_fullDR": settled.round(3).tolist(),
      "settled_distribution_1024_envs (see settle study)": {
        "no_DR": {"supine": 1.0},
        "full_DR": {
          "supine": 0.363,
          "upright_gate": 0.173,
          "side": 0.104,
          "prone": 0.005,
        },
        "main_cause": "actuation_offset (+-1 N m constant torque while unactuated)",
      },
    },
  )

  with open(os.path.join(args.out, "validation.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
  print(f"\n{sum(r['ok'] for r in RESULTS.values())}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
  main()
