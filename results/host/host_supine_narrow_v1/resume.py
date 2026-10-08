"""Resume the isolated narrow asset; never launch this file automatically."""
import argparse
import csv
import json
import subprocess
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco
import torch

import experiment as experiment
from minipi_getup.host.config import PiCfg, PiCfgPPO, class_to_dict
from minipi_getup.host_rl.on_policy_runner import OnPolicyRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=experiment.OUT/'logs/model_460_final.pt')
    p.add_argument('--max-iterations', type=int, default=8000, help='Total iteration count, including resumed updates')
    p.add_argument('--num-envs', type=int, default=4096)
    args = p.parse_args()
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
    if busy:
        raise SystemExit(f'GPU already has a compute process; refusing to start: {busy}')
    experiment.use_asset()
    experiment.seed(1)
    cfg, pcfg = PiCfg(), PiCfgPPO()
    pcfg.runner.save_interval = 50
    pcfg.runner.run_name = 'host_supine_narrow_v1_resume'
    env = experiment.hm.LeggedRobot_Pi(cfg, args.num_envs, 'cuda:0')
    expected = mujoco.MjModel.from_xml_path(str(experiment.ASSET)).jnt_range[1:]
    assert np.allclose(env.robot.data.joint_pos_limits[0].cpu().numpy(), expected)
    assert env.joint_vel_cap is None and env.limit_solref is None
    logdir = experiment.OUT / ('resume_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    logdir.mkdir()
    runner = OnPolicyRunner(env, cfg, class_to_dict(pcfg), str(logdir), device='cuda:0')
    infos = runner.load(str(args.checkpoint)) or {}
    if args.max_iterations <= runner.current_learning_iteration:
        raise SystemExit('--max-iterations must exceed the checkpoint iteration')
    if 'curriculum_state' in infos:
        state = infos['curriculum_state']
        for name in ('force', 'action_rescale'):
            saved = state[name].to(env.device)
            target = getattr(env, name)
            target.copy_(saved if saved.shape == target.shape else saved.mean().expand_as(target))
        note = 'Restored saved curriculum tensors (means if env count changed).'
    else:
        if args.checkpoint.resolve() != (experiment.OUT/'logs/model_460_final.pt').resolve():
            raise SystemExit('Checkpoint has no curriculum state; use the original model_460_final.pt first.')
        with (experiment.OUT/'training.csv').open() as f:
            last = list(csv.DictReader(f))[-1]
        env.force.fill_(float(last['pull_force']))
        env.action_rescale.fill_(float(last['action_scale']))
        note = 'Original checkpoint lacks per-environment curriculum: initialized from last logged population means.'
    env.compute_observations()
    old_save = runner.save
    def save(self, path, infos=None, it=None):
        payload = dict(infos or {})
        payload['curriculum_state'] = {name: getattr(env, name).detach().cpu() for name in ('force', 'action_rescale')}
        old_save(path, infos=payload, it=it)
    runner.save = types.MethodType(save, runner)
    completed = [runner.current_learning_iteration]
    old_log = runner.log
    def log(self, locs):
        completed[0] = locs['it'] + 1
        old_log(locs)
    runner.log = types.MethodType(log, runner)
    (logdir/'config.json').write_text(json.dumps({'checkpoint': str(args.checkpoint), 'asset': str(experiment.ASSET), 'env_cfg': class_to_dict(cfg), 'train_cfg': class_to_dict(pcfg), 'num_envs': args.num_envs, 'max_iterations': args.max_iterations, 'curriculum_resume': note}, indent=2))
    print(f'Resuming iteration {runner.current_learning_iteration}; logs: {logdir}\n{note}', flush=True)
    try:
        runner.learn(args.max_iterations-runner.current_learning_iteration, init_at_random_ep_len=pcfg.runner.init_at_random_ep_len)
    except KeyboardInterrupt:
        runner.current_learning_iteration = completed[0]
        runner.save(str(logdir/'model_interrupted.pt'), it=completed[0])
        print('Interrupted checkpoint saved; an interruption mid-update is not an exact completed-update boundary.', flush=True)
    finally:
        if runner.writer:
            runner.writer.flush()
            runner.writer.close()


if __name__ == '__main__':
    main()
