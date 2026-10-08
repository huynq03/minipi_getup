"""Isolated HoST training on asset-zoo cl_pai; scratch by default.

--plant real (default): the real robot's PD gains and torque limits instead of HoST's.
  Gains: vendor RL package clpai_12dof_0905 devel_config running_kp / running_kd.
  Torque limits: pai_control pai_model.yaml tau_limits (16 N m, ankle roll 6 N m).
--plant host: HoST's gains (30/15/15/30/12/5, kd 0.2) and 20 N m on every joint.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import types

REAL_KP = {'hip_pitch': 60.0, 'hip_roll': 40.0, 'thigh': 20.0, 'calf': 60.0, 'ankle_pitch': 30.0, 'ankle_roll': 10.0}
REAL_KD = {'hip_pitch': 2.4, 'hip_roll': 0.8, 'thigh': 0.4, 'calf': 2.8, 'ankle_pitch': 1.6, 'ankle_roll': 0.3}
REAL_TORQUE = {'hip_pitch': 16.0, 'hip_roll': 16.0, 'thigh': 16.0, 'calf': 16.0, 'ankle_pitch': 16.0, 'ankle_roll': 6.0}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--num-envs', type=int, default=4096)
  parser.add_argument('--max-iterations', type=int, default=8000, help='Total PPO updates including resumed updates')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--resume', type=Path, help='Only checkpoints from this cl_pai launcher')
  parser.add_argument('--plant', choices=('real', 'host'), default='real', help='PD gains and torque limits')
  args = parser.parse_args()
  if args.num_envs < 1 or args.max_iterations < 1:
    parser.error('Environment and iteration counts must be positive.')
  busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
  if busy:
    parser.error(f'GPU has active compute processes; not starting: {busy}')

  import warp as wp
  wp.config.kernel_cache_dir = '/tmp/host-supine-narrow-warp'
  # Package registration may already have initialized Warp before main().
  from warp._src.build import init_kernel_cache
  init_kernel_cache('/tmp/host-supine-narrow-warp')
  import mujoco
  import numpy as np
  import torch
  from minipi_getup.host.config import PiCfg, PiCfgPPO, class_to_dict
  from minipi_getup.host.train import set_seed
  from minipi_getup.host.clpai_asset import use_clpai_asset
  import minipi_getup.host.env as host_env
  from minipi_getup.host.env import LeggedRobot_Pi
  from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner

  torch.set_num_threads(4)
  set_seed(args.seed)
  asset = use_clpai_asset()
  asset_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
  if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    info = checkpoint.get('infos') or {}
    if info.get('asset_sha256') != asset_hash or info.get('asset_id') != 'cl_pai':
      parser.error('Resume checkpoint does not match this cl_pai asset. Start scratch or select a cl_pai checkpoint.')
    if info.get('plant', 'host') != args.plant:
      parser.error(f'Resume checkpoint was trained with --plant {info.get("plant", "host")}.')
    del checkpoint
  cfg, pcfg = PiCfg(), PiCfgPPO()
  cfg.asset.file = str(asset)
  pcfg.seed = args.seed
  pcfg.runner.max_iterations = args.max_iterations
  pcfg.runner.save_interval = 50
  pcfg.runner.run_name = 'host_clpai_narrow_v1' if args.plant == 'host' else 'host_clpai_realpd_v1'
  if args.plant == 'real':
    cfg.control.stiffness, cfg.control.damping = dict(REAL_KP), dict(REAL_KD)
    host_env.TORQUE_LIMITS = dict(REAL_TORQUE)
  env = LeggedRobot_Pi(cfg, args.num_envs, 'cuda:0')
  native = mujoco.MjModel.from_xml_path(str(asset))
  expected = [mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, native.njnt)]
  assert env.dof_names == expected
  assert np.allclose(env.robot.data.joint_pos_limits[0].cpu().numpy(), native.jnt_range[1:])
  assert len(env.head_indices) == 1 and len(env.left_ankle_indices) == len(env.right_ankle_indices) == 5
  assert env.joint_vel_cap is None and env.limit_solref is None
  limits = REAL_TORQUE if args.plant == 'real' else None
  for i, name in enumerate(env.dof_names):
    kind = next(k for k in REAL_KP if name.endswith(f'_{k}_joint'))
    assert np.isclose(float(env.p_gains[i]), cfg.control.stiffness[kind]) and np.isclose(float(env.d_gains[i]), cfg.control.damping[kind])
    assert np.isclose(float(env.torque_limits[i]), limits[kind] if limits else host_env.URDF_EFFORT)
  # ctrl i must drive dof i (torques are written as one block) with dof i's limit.
  mj = env.sim.mj_model
  ctrl_joints = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j).split('/')[-1] for j in mj.actuator_trnid[:, 0]]
  assert ctrl_joints == env.dof_names, ctrl_joints
  assert np.allclose(mj.actuator_forcerange[:, 1], env.torque_limits.cpu().numpy()), mj.actuator_forcerange
  logdir = Path.cwd()/'results/host'/pcfg.runner.run_name/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
  logdir.mkdir(parents=True)
  (logdir/'config.json').write_text(json.dumps({
    'asset': str(asset), 'asset_sha256': asset_hash, 'seed': args.seed,
    'num_envs': args.num_envs, 'max_iterations': args.max_iterations,
    'resume': str(args.resume) if args.resume else None, 'plant': args.plant,
    'torque_limits': dict(zip(env.dof_names, env.torque_limits.tolist())),
    'reference_frames': '9 welded massless geometry-free HoST measurement frames added in memory',
    'env_cfg': class_to_dict(cfg), 'train_cfg': class_to_dict(pcfg),
  }, indent=2))
  runner = OnPolicyRunner(env, cfg, class_to_dict(pcfg), str(logdir), device='cuda:0')
  if args.resume:
    state = runner.load(str(args.resume))['curriculum_state']
    for name in ('force', 'action_rescale'):
      target = getattr(env, name); saved = state[name].to(env.device)
      target.copy_(saved if saved.shape == target.shape else saved.mean().expand_as(target))
    env.compute_observations()
  if args.max_iterations <= runner.current_learning_iteration:
    parser.error('--max-iterations must exceed resumed iteration.')
  old_save, old_log = runner.save, runner.log
  completed = [runner.current_learning_iteration]

  def save(self, path, infos=None, it=None):
    payload = dict(infos or {})
    payload.update(asset_id='cl_pai', asset_sha256=asset_hash, plant=args.plant,
      curriculum_state={n: getattr(env, n).detach().cpu() for n in ('force', 'action_rescale')})
    old_save(path, infos=payload, it=it)

  def log(self, locs):
    completed[0] = locs['it'] + 1
    old_log(locs)

  runner.save, runner.log = types.MethodType(save, runner), types.MethodType(log, runner)
  print(f'HoST cl_pai: {"resume" if args.resume else "scratch"}; plant={args.plant}; '
        f'kp={env.p_gains.tolist()} kd={env.d_gains.tolist()} tau={env.torque_limits.tolist()}; '
        f'asset={asset}; logs={logdir}', flush=True)
  try:
    runner.learn(args.max_iterations-runner.current_learning_iteration,
                 init_at_random_ep_len=pcfg.runner.init_at_random_ep_len)
  except KeyboardInterrupt:
    runner.current_learning_iteration = completed[0]
    runner.save(str(logdir/'model_interrupted.pt'), it=completed[0])
    print('Interrupted checkpoint saved (may include a partial PPO update).', flush=True)
  finally:
    if runner.writer:
      runner.writer.flush(); runner.writer.close()


if __name__ == '__main__':
  main()
