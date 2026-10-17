# Adapted from HoST rsl_rl/rsl_rl/runners/on_policy_runner.py.
#
# The construction, rollout, ``process_env_step`` / ``compute_returns`` / ``update`` call
# sequence, ``init_at_random_ep_len`` and the save schedule are unchanged. Additions are
# marked "PORT:" and only observe: extra TensorBoard scalars, non-finite checks that stop
# training with a diagnostic dump, and checkpoints that also record the real iteration
# (the release saves ``current_learning_iteration``, which is the *start* iteration
# during a run, so its checkpoints cannot be resumed at the right iteration).
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin. BSD-3-Clause (see the HoST release).

import copy
import json
import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from minipi_getup.host_rl.actor_critic import ActorCritic
from minipi_getup.host_rl.ppo import PPO, NonFiniteError


class OnPolicyRunner:

    def __init__(self,
                 env,
                 env_cfg,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.num_critics = env_cfg.rewards.num_reward_groups
        self.reward_group_weights = env_cfg.rewards.reward_group_weights
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        self.num_critics,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, self.reward_group_weights, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], self.num_critics)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_saved = None  # PORT

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self.it = it  # PORT
            mon = _StepMonitor(self.env, self.device)  # PORT
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    try:  # PORT
                        mon.step(obs, rewards, dones)
                    except NonFiniteError as e:
                        self._dump_failure(f"rollout step {i}: {e.what}", e.info, mon)
                        raise
                    self.alg.process_env_step(rewards, dones, infos)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards.sum(-1)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            mon.check_storage(self.alg.storage, self)  # PORT

            # PORT: snapshot the pre-update state so a failing update can be dumped.
            pre_update = (copy.deepcopy(self.alg.actor_critic.state_dict()), copy.deepcopy(self.alg.optimizer.state_dict()))
            try:
                mean_value_loss, mean_surrogate_loss = self.alg.update()
            except NonFiniteError as e:
                self._dump_failure(f"update: {e.what}", e.info, mon, pre_update)
                raise
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it=it)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), it=self.current_learning_iteration)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device).flatten()))
                value = torch.mean(infotensor.float())
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        # PORT: extra scalars.
        port = locs['mon'].summary()
        for k, v in self.alg.stats.items():
            port['PPO/' + k] = v
        port['Policy/min_noise_std'] = self.alg.actor_critic.std.min().item()
        port['Policy/max_noise_std'] = self.alg.actor_critic.std.max().item()
        for k, v in port.items():
            self.writer.add_scalar(k, v, locs['it'])
        self.last_port_metrics = port

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                        'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        if len(locs['rewbuffer']) > 0:
            log_string += (f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                           f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")

        log_string += ep_string
        for k in sorted(port):  # PORT
            log_string += f"""{k + ':':>{pad}} {port[k]:.4f}\n"""
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - self.start_it + 1) * (
                               locs['num_learning_iterations'] - (locs['it'] - self.start_it)):.1f}s\n""")
        print(log_string, flush=True)

    @property
    def start_it(self):  # PORT
        return self.current_learning_iteration

    def save(self, path, infos=None, it=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'it': it,  # PORT: the real iteration
            'learning_rate': self.alg.learning_rate,  # PORT
            'infos': infos,
            }, path)
        self.last_saved = path

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        if loaded_dict.get('it') is not None:  # PORT
            self.current_learning_iteration = loaded_dict['it']
        if loaded_dict.get('learning_rate') is not None:  # PORT: adaptive lr state
            self.alg.learning_rate = loaded_dict['learning_rate']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    # PORT: failure dump.
    def _dump_failure(self, reason, info, mon, pre_update=None):
        it = getattr(self, 'it', -1)
        d = os.path.join(self.log_dir, f'nan_failure_it{it}')
        os.makedirs(d, exist_ok=True)
        if pre_update is not None:
            torch.save({'model_state_dict': pre_update[0], 'optimizer_state_dict': pre_update[1], 'it': it}, os.path.join(d, 'pre_update.pt'))
        env = self.env
        state = {k: getattr(env, k).detach().cpu() for k in ('root_states', 'dof_pos', 'dof_vel', 'torques', 'actions', 'action_rescale', 'force', 'obs_buf', 'rew_buf') if hasattr(env, k)}
        torch.save(state, os.path.join(d, 'env_state.pt'))
        with open(os.path.join(d, 'report.json'), 'w') as f:
            json.dump({'reason': reason, 'iteration': it, 'info': info, 'last_valid_checkpoint': self.last_saved,
                       'metrics_before_failure': getattr(self, 'last_port_metrics', {}),
                       'this_iteration_partial': mon.summary(partial=True)}, f, indent=2, default=str)
        print(f'[HoST-port] NON-FINITE: {reason} at iteration {it}; dump in {d}; last valid checkpoint {self.last_saved}', flush=True)


class _StepMonitor:
    """PORT: per-iteration statistics gathered during the rollout (no effect on training)."""

    def __init__(self, env, device):
        self.env = env
        self.n = 0
        self.acc = {}
        self.resets = {}
        self.reset_count = 0
        self.nonfinite = {'obs': 0, 'rew': 0}
        self.device = device

    def _add(self, k, v):
        self.acc[k] = self.acc.get(k, 0.0) + v

    def step(self, obs, rewards, dones):
        env = self.env
        bad_obs = (~torch.isfinite(obs)).sum()
        bad_rew = (~torch.isfinite(rewards)).sum()
        vals = torch.stack([
            bad_obs.float(), bad_rew.float(),
            env.force.mean(), (env.force > 0).float().mean(),
            env.action_rescale.mean(), env.action_rescale.min(), env.action_rescale.max(),
            env.torques.abs().mean(), env.torques.abs().max(),
            env.dof_vel.abs().mean(), env.dof_vel.abs().max(),
            env.old_headheight.mean(), env.old_headheight.max(),
            env.root_states[:, 2].mean(),
            (env.pending_force[:, 0, 2] > 0).float().mean(),
            *[rewards[:, i].mean() for i in range(rewards.shape[1])],
            *[v.float().mean() for v in env.term_values.values()],
        ]).cpu().tolist()
        names = ['nonfinite_obs', 'nonfinite_rew', 'Curriculum/force_mean', 'Curriculum/force_pos_frac',
                 'Curriculum/action_rescale_mean', 'Curriculum/action_rescale_min', 'Curriculum/action_rescale_max',
                 'Robot/torque_abs_mean', 'Robot/torque_abs_max', 'Robot/joint_vel_abs_mean', 'Robot/joint_vel_abs_max',
                 'Robot/head_height_mean', 'Robot/head_height_max', 'Robot/base_height_mean', 'Curriculum/force_applied_frac']
        names += [f'RewardGroup/{g}' for g in env.reward_groups]
        names += [f'RewardTerm/{k}' for k in env.term_values.keys()]
        for k, v in zip(names, vals):
            if k.startswith('nonfinite'):
                self.nonfinite[k.split('_')[1]] += int(v)
            elif k.endswith('_max'):
                self.acc[k] = max(self.acc.get(k, -1e30), v)
            elif k.endswith('_min'):
                self.acc[k] = min(self.acc.get(k, 1e30), v)
            else:
                self._add(k, v)
        self.n += 1
        if env.reset_stats and env.reset_stats_step == env.common_step_counter:
            nr = int(dones.sum().item())
            for k, v in env.reset_stats.items():
                if k == 'per_joint_peak_torque':
                    self.resets[k] = torch.maximum(self.resets.get(k, torch.zeros_like(v)), v)
                    continue
                v = float(v)
                if v == v:  # skip nan (e.g. time_to_stand with no stander)
                    self.resets.setdefault(k, [0.0, 0])
                    self.resets[k][0] += v * nr
                    self.resets[k][1] += nr
        if self.nonfinite['obs'] or self.nonfinite['rew']:
            raise NonFiniteError('rollout obs/reward', dict(self.nonfinite))

    def check_storage(self, storage, runner):
        t = {'advantages': storage.advantages, 'returns': storage.returns, 'values': storage.values,
             'combined_advantages': storage.multi_critic_advantages}
        for k, v in t.items():
            if not torch.isfinite(v).all():
                runner._dump_failure(f'storage {k}', {k: int((~torch.isfinite(v)).sum())}, self)
                raise NonFiniteError(f'storage {k}', {})

    def summary(self, partial=False):
        out = {}
        for k, v in self.acc.items():
            out[k] = v if (k.endswith('_max') or k.endswith('_min')) else v / max(self.n, 1)
        for k, (s, c) in ((k, v) for k, v in self.resets.items() if k != 'per_joint_peak_torque'):
            out[f'Episode_end/{k}'] = s / max(c, 1)
        if 'per_joint_peak_torque' in self.resets:
            for name, v in zip(self.env.dof_names, self.resets['per_joint_peak_torque'].tolist()):
                out[f'PeakTorque/{name}'] = v
        out['Diag/nonfinite_obs'] = self.nonfinite['obs']
        out['Diag/nonfinite_rew'] = self.nonfinite['rew']
        return out
