"""FTSR runner (paper Alg. 1; release OnPolicyRunner + TSPPO + EncoderMSE).

Per iteration:

1. Rollout, ``num_steps_per_env`` steps. Rows ``[0, num_teacher)`` are the teacher
   group (act on z = E^t(x_t)), the others the student group (z = E^s(o_{t-4..t})).
   Latents come from the same step as o_t (release lag Q3 fixed). Critic input:
   ``[E^t(x_t), s_t]`` for every row (release). Time-outs bootstrap
   ``r += gamma V`` (release ``process_env_step``).
2. Constraint costs: the normalized Eq. 4 force / torque of each step
   (``mdp.actions``), from ``env.extras``.
3. GAE and Eq. 8 (``storage.py``).
4. PPO update: teacher rows get a fresh E^t latent with gradient into E^t, student
   rows their stored latent (Eq. 6: j = s updates theta only); the critic uses the
   stored teacher latent (release: the latent is part of the stored critic obs).
   Teacher identity is a per-sample mask (release bug Q1 fixed). One Adam optimizer and
   one gradient clip over actor, std, critic and E^t (release).
5. Student encoder: MSE(E^s(o), E^t(x).detach()) on all samples, with the updated E^t
   (release EncoderMSE, paper Eq. 10).

Physics explosions (non-finite state in one env): the env is flagged by the action
term (wrench and costs zeroed at the source) and by mjlab's ``nan`` termination, which
resets it. Its sample is stored as invalid: the trajectory is truncated there, it gets
no advantage and no PPO loss weight, and its episode is not counted in the reward
statistics (``storage.py``). Each occurrence is printed and logged (``Numerics/``);
frequent explosions stop training (``explosion_*``).

Checkpoints hold everything needed to continue: networks, both optimizers, adaptive
learning rate, iteration, env step counter (assist schedule) and RNG states. The stage
is stateless (recomputed from the population).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict, deque

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from minipi_getup.ftsr_ref.mdp.actions import COST_KEY, INVALID_KEY, QD_STEP_MAX_KEY
from minipi_getup.ftsr_ref.mdp.stages import STAGE_KEY
from minipi_getup.ftsr_ref.rl.modules import FtsrModel, StudentPolicy
from minipi_getup.ftsr_ref.rl.storage import RolloutStorage

GROUPS = ("actor", "policy", "teacher", "critic")

# Terminal names of the reward terms (``Episode_Reward/<term>``, per-second episode
# averages from the reward manager), in print order: tracking terms first.
REWARD_SHORT = {
  "track_base_height_exp": "height",
  "track_lin_vel_xy_exp": "lin_vel",
  "track_ang_vel_yaw_exp": "yaw",
  "rew_wheel_contact_force": "contact_force",
  "pen_base_orientation_l2": "orient",
  "pen_base_orientation_z_l2": "orient_z",
  "pen_no_fly_l2": "no_fly",
  "pen_action_rate_l2": "act_rate",
  "pen_action_smoothness_l2": "act_smooth",
  "pen_dof_pos_bias_l2": "pos_bias",
  "pen_two_leg_bias_l2": "leg_bias",
  "pen_feet_distance_l2": "feet_dist",
  "pen_dof_pos_limits": "pos_limits",
  "pen_dof_vel_l2": "dof_vel",
  "pen_dof_acc_l2": "dof_acc",
  "pen_max_velocity_l2": "max_vel",
  "qd_soft_envelope": "qd_envelope",
  "pen_torques_l2": "torque",
  "pen_torque_limits": "torque_limits",
  "pen_joint_power_l2": "power",
  "pen_termination": "termination",
}


class FtsrRunner:
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.writer: SummaryWriter | None = None
    self.current_learning_iteration = 0
    torch.manual_seed(train_cfg["seed"])

    obs = env.get_observations()
    self.dims = {g: obs[g].shape[-1] for g in GROUPS}
    self.num_envs = env.num_envs
    self.num_actions = env.num_actions
    self.num_teacher = int(round(train_cfg["teacher_fraction"] * self.num_envs))
    self.teacher_mask = torch.arange(self.num_envs, device=device) < self.num_teacher

    self.model = FtsrModel(
      self.dims,
      self.num_actions,
      latent_dim=train_cfg["latent_dim"],
      actor_hidden=tuple(train_cfg["actor_hidden_dims"]),
      critic_hidden=tuple(train_cfg["critic_hidden_dims"]),
      encoder_hidden=tuple(train_cfg["encoder_hidden_dims"]),
      init_noise_std=train_cfg["init_noise_std"],
    ).to(device)
    self.lr = train_cfg["learning_rate"]
    self.optimizer = torch.optim.Adam(self.model.ppo_parameters(), lr=self.lr)
    self.student_optimizer = torch.optim.Adam(
      self.model.student_encoder.parameters(), lr=train_cfg["student_learning_rate"]
    )
    self.storage = RolloutStorage(
      train_cfg["num_steps_per_env"],
      self.num_envs,
      self.dims,
      self.num_actions,
      train_cfg["latent_dim"],
      num_costs=2,
      device=device,
    )
    self._eval_procs: list[subprocess.Popen] = []
    print(
      f"[FTSR-Ref] envs={self.num_envs} teacher={self.num_teacher} "
      f"student={self.num_envs - self.num_teacher} dims={self.dims}",
      flush=True,
    )
    init = train_cfg.get("init_checkpoint") or ""
    if init:
      self.load_weights(init)

  # ---------------------------------------------------------------------------
  # Acting.

  def act(self, obs, deterministic: bool = False):
    m = self.model
    z_t = m.teacher_encoder(obs["teacher"])
    z_s = m.student_encoder(obs["policy"])
    z = torch.where(self.teacher_mask.unsqueeze(-1), z_t, z_s)
    mean = m.actor_mean(z, obs["actor"])
    std = m.std.clamp(min=1e-4).expand_as(mean)
    value = m.value(z_t, obs["critic"])
    return z_t, z_s, mean, std, value

  def get_inference_policy(self, device=None, mode: str = "student"):
    """Deterministic policy on a TensorDict of observation groups. ``student`` is the
    deployable path (``StudentPolicy`` on the ``policy`` group)."""
    self.model.eval()
    if device is not None:
      self.model.to(device)
    model = self.model
    student = StudentPolicy(model)

    def policy(obs):
      with torch.no_grad():
        if mode == "teacher":
          z = model.teacher_encoder(obs["teacher"])
          return model.actor_mean(z, obs["actor"])
        return student(obs["policy"])

    return policy

  # ---------------------------------------------------------------------------
  # Training.

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    cfg = self.cfg
    if self.log_dir is not None and self.writer is None:
      os.makedirs(self.log_dir, exist_ok=True)
      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    unwrapped = self.env.unwrapped
    if init_at_random_ep_len:
      unwrapped.episode_length_buf = torch.randint_like(
        unwrapped.episode_length_buf, high=int(unwrapped.max_episode_length)
      )
    action_term = unwrapped.action_manager.get_term("joint_pos")
    monitor = action_term.monitor

    obs = self.env.get_observations().to(self.device)
    rew_buf: deque[float] = deque(maxlen=200)
    len_buf: deque[float] = deque(maxlen=200)
    cur_rew = torch.zeros(self.num_envs, device=self.device)
    cur_len = torch.zeros(self.num_envs, device=self.device)
    gamma, lam = cfg["gamma"], cfg["lam"]
    start = self.current_learning_iteration
    end = start + num_learning_iterations

    T = cfg["num_steps_per_env"]
    tm = unwrapped.termination_manager
    if "nan" in tm.active_terms:

      def nan_term():
        return tm.get_term("nan")
    else:
      zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

      def nan_term():
        return zero

    robot = unwrapped.scene["robot"]

    def base_h():  # base height before the step (diagnostics of explosions)
      return robot.data.root_link_pos_w[:, 2].clone()

    h_before = base_h()
    for it in range(start, end):
      t0 = time.time()
      self.model.train()
      # Rollout logs stay on the device (sum, count); reduced once per iteration.
      log_sum: dict[str, torch.Tensor | float] = {}
      log_cnt: dict[str, int] = defaultdict(int)
      monitor.clear()
      cost_seen = torch.zeros((), dtype=torch.bool, device=self.device)
      done_mask = torch.zeros(T, self.num_envs, dtype=torch.bool, device=self.device)
      done_rew = torch.zeros(T, self.num_envs, device=self.device)
      done_len = torch.zeros(T, self.num_envs, device=self.device)
      inv_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      inv_qd = torch.zeros(self.num_envs, device=self.device)
      inv_h = torch.full((self.num_envs,), float("nan"), device=self.device)
      with torch.inference_mode():
        for step in range(T):
          z_t, z_s, mean, std, value = self.act(obs)
          dist = torch.distributions.Normal(mean, std)
          actions = dist.sample()
          step_obs = {g: obs[g] for g in GROUPS}
          obs, rewards, dones, extras = self.env.step(actions)
          obs = obs.to(self.device)
          cost = unwrapped.extras[COST_KEY].clone()
          inv = unwrapped.extras[INVALID_KEY] | nan_term()
          inv_any |= inv
          inv_qd = torch.where(inv, unwrapped.extras[QD_STEP_MAX_KEY], inv_qd)
          inv_h = torch.where(inv, h_before, inv_h)
          cost_seen |= (cost.abs() > 0.0).any()
          rewards = rewards.clone()
          if "time_outs" in extras:
            rewards += gamma * value * extras["time_outs"].float()
          self.storage.add(
            step_obs,
            z_t=z_t,
            z_s=z_s,
            actions=actions,
            log_prob=dist.log_prob(actions).sum(-1),
            mu=mean,
            sigma=std,
            rewards=rewards,
            costs=cost,
            dones=dones.float(),
            values=value,
            valid=~inv,
          )
          h_before = base_h()
          items = dict(extras.get("log", {}))
          if action_term.cfg.assist is not None:
            items.update(action_term.log_assist())
          for k, v in items.items():
            v = v.float().mean() if torch.is_tensor(v) else float(v)
            log_sum[k] = log_sum[k] + v if k in log_sum else v
            log_cnt[k] += 1
          cur_rew += rewards
          cur_len += 1
          d = dones > 0
          done_mask[step] = d & ~inv
          done_rew[step] = cur_rew
          done_len[step] = cur_len
          cur_rew *= ~d
          cur_len *= ~d

      # Finished episodes, in the same (step, env) order as a per-step nonzero().
      rew_buf.extend(done_rew[done_mask].tolist())
      len_buf.extend(done_len[done_mask].tolist())
      logs = {
        k: [float(v) / log_cnt[k]] for k, v in log_sum.items()
      }  # one host transfer per key
      with torch.no_grad():
        z_t = self.model.teacher_encoder(obs["teacher"])
        last_value = self.model.value(z_t, obs["critic"])
        adv_info = self.storage.compute(
          last_value,
          gamma,
          lam,
          tuple(cfg["penalty_factors"]),
          cfg["use_force_guidance"] and bool(cost_seen),
          cfg["cost_j_mode"],
          cfg["standardize_cost_advantage"],
        )
      t_collect = time.time() - t0
      t1 = time.time()
      loss_info = self._update()
      loss_info.update(self._update_student())
      self.storage.clear()
      t_learn = time.time() - t1
      self.current_learning_iteration = it + 1

      safety = monitor.summary(action_term.joint_names)
      self._explosions(it, inv_any, inv_qd, inv_h, adv_info, loss_info)
      self._log(
        it, logs, loss_info, adv_info, safety, rew_buf, len_buf, t_collect, t_learn
      )
      self._report_transitions(it, logs)
      if self.log_dir is not None:
        if (it + 1) % cfg["save_interval"] == 0 or it == end - 1:
          self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
        if self._is_eval_iteration(it + 1):
          path = os.path.join(self.log_dir, f"model_{it + 1}.pt")
          if not os.path.exists(path):
            self.save(path)
          self._spawn_eval(path, it + 1)
    if self.writer is not None:
      self.writer.flush()

  # Non-finite guard (diagnostics only; the objective is unchanged while values are
  # finite): a minibatch whose loss or gradient norm is non-finite is skipped (no
  # optimizer step) and its inputs are dumped once per iteration. Persistent failures
  # stop training: more than half of an iteration's minibatches skipped, or skips in
  # ``nan_max_consecutive`` consecutive iterations.
  nan_max_consecutive = 3
  # Physics explosions: stop (and report) when more than ``explosion_max_envs`` envs
  # explode in one iteration, or explosions occur in more than
  # ``explosion_max_iters`` of the last ``explosion_window`` iterations.
  explosion_max_envs = 8
  explosion_window = 100
  explosion_max_iters = 5

  def _explosions(self, it, inv_any, inv_qd, inv_h, adv_info, loss_info) -> None:
    ids = inv_any.nonzero().squeeze(-1)
    n = int(ids.numel())
    hist = getattr(self, "_explosion_hist", None)
    if hist is None:
      hist = self._explosion_hist = deque(maxlen=self.explosion_window)
    hist.append(n > 0)
    qd = inv_qd[ids]
    rec = {
      "invalid_envs": float(n),
      "invalid_samples": adv_info.get("invalid_samples", 0.0),
      "nonfinite_cost_samples": adv_info.get("nonfinite_cost_samples", 0.0),
      "invalid_qd_max_before": float(qd.max()) if n else 0.0,
      "explosion_iters_in_window": float(sum(hist)),
    }
    loss_info.update({"numerics_" + k: v for k, v in rec.items()})
    if n == 0:
      return
    info = {
      "iteration": it,
      **rec,
      "env_ids": ids.tolist()[:32],
      "qd_max_before_per_env": [round(float(x), 2) for x in qd[:32]],
      "base_height_before_per_env": [round(float(x), 3) for x in inv_h[ids][:32]],
      "skipped_minibatches": loss_info.get("skipped_minibatches", 0.0),
    }
    print(f"[FTSR-Ref] PHYSICS EXPLOSION: {json.dumps(info)}", flush=True)
    if self.log_dir is not None:
      with open(os.path.join(self.log_dir, "explosions.jsonl"), "a") as f:
        f.write(json.dumps(info) + "\n")
    if n > self.explosion_max_envs or sum(hist) > self.explosion_max_iters:
      raise RuntimeError(
        f"frequent physics explosions ({n} envs at iteration {it}; "
        f"{sum(hist)} of the last {len(hist)} iterations): stopping; see "
        "explosions.jsonl; last checkpoint is intact"
      )

  def _nonfinite_report(self, it_tag: str, tensors: dict[str, torch.Tensor]) -> None:
    rep = {}
    for k, v in tensors.items():
      v = v.detach()
      bad = ~torch.isfinite(v)
      fin = v[~bad]
      rep[k] = {
        "nonfinite": int(bad.sum()),
        "numel": v.numel(),
        "absmax_finite": float(fin.abs().max()) if fin.numel() else None,
      }
    rep["params_nonfinite"] = {
      n: int((~torch.isfinite(p)).sum())
      for n, p in self.model.named_parameters()
      if not torch.isfinite(p).all()
    }
    print(f"[FTSR-Ref] NON-FINITE in {it_tag}: {json.dumps(rep)}", flush=True)
    if self.log_dir is not None:
      with open(os.path.join(self.log_dir, f"nonfinite_{it_tag}.json"), "w") as f:
        json.dump(rep, f, indent=1)

  def _update(self) -> dict[str, float]:
    cfg = self.cfg
    clip = cfg["clip_param"]
    stats: dict[str, float | torch.Tensor] = defaultdict(float)
    n = 0
    skipped, total_mb, reported = 0, 0, False
    masked = self.storage.has_invalid
    for mb in self.storage.minibatches(
      cfg["num_mini_batches"], cfg["num_learning_epochs"], self.teacher_mask
    ):
      m = self.model
      z_t_new = m.teacher_encoder(mb["obs_teacher"])
      z = torch.where(mb["teacher"].unsqueeze(-1), z_t_new, mb["z_s"])
      mean = m.actor_mean(z, mb["obs_actor"])
      std = m.std.clamp(min=1e-4).expand_as(mean)
      dist = torch.distributions.Normal(mean, std)
      log_prob = dist.log_prob(mb["actions"]).sum(-1)
      value = m.value(mb["z_t"], mb["obs_critic"])
      if masked:
        # Invalid samples (physics explosions) carry no weight.
        w = mb["valid"]
        n_w = w.sum().clamp(min=1)

        def avg(x, w=w, n_w=n_w):
          return torch.where(w, x, 0.0).sum() / n_w
      else:
        avg = torch.mean
      entropy = avg(dist.entropy().sum(-1))

      if cfg["schedule"] == "adaptive" and cfg["desired_kl"] > 0:
        with torch.no_grad():
          old_mu, old_sigma = mb["mu"], mb["sigma"]
          kl = torch.sum(
            torch.log(std / old_sigma + 1e-5)
            + (old_sigma**2 + (old_mu - mean) ** 2) / (2.0 * std**2)
            - 0.5,
            dim=-1,
          )
          kl = avg(kl)
          if kl > cfg["desired_kl"] * 2.0:
            self.lr = max(1e-5, self.lr / 1.5)
          elif 0.0 < kl < cfg["desired_kl"] / 2.0:
            self.lr = min(1e-2, self.lr * 1.5)
          for group in self.optimizer.param_groups:
            group["lr"] = self.lr
          stats["kl"] += float(kl)

      adv = mb["advantages"]
      ratio = torch.exp(log_prob - mb["log_prob"])
      surrogate = avg(
        torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip))
      )
      if cfg["use_clipped_value_loss"]:
        v_clip = mb["values"] + (value - mb["values"]).clamp(-clip, clip)
        value_loss = avg(
          torch.max((value - mb["returns"]) ** 2, (v_clip - mb["returns"]) ** 2)
        )
      else:
        value_loss = avg((value - mb["returns"]) ** 2)
      loss = (
        surrogate
        + cfg["value_loss_coef"] * value_loss
        - cfg["entropy_coef"] * (entropy)
      )
      self.optimizer.zero_grad()
      total_mb += 1
      ok = bool(torch.isfinite(loss))
      if ok:
        loss.backward()
        grad = nn.utils.clip_grad_norm_(m.ppo_parameters(), cfg["max_grad_norm"])
        ok = bool(torch.isfinite(grad))
      if not ok:
        self.optimizer.zero_grad()
        skipped += 1
        if not reported:
          reported = True
          self._nonfinite_report(
            f"ppo_it{self.current_learning_iteration}",
            {
              "loss": loss,
              "surrogate": surrogate,
              "value_loss": value_loss,
              "mean": mean,
              "std": m.std,
              "log_prob": log_prob,
              "old_log_prob": mb["log_prob"],
              "ratio": ratio,
              "advantages": adv,
              "returns": mb["returns"],
              "values": mb["values"],
              "value": value,
              "actions": mb["actions"],
              "obs_actor": mb["obs_actor"],
              "obs_critic": mb["obs_critic"],
              "obs_teacher": mb["obs_teacher"],
            },
          )
        continue
      self.optimizer.step()
      with torch.no_grad():  # device accumulators, read once after the update
        stats["surrogate"] += surrogate.detach()
        stats["value"] += value_loss.detach()
        stats["entropy"] += entropy.detach()
        stats["grad_norm"] += grad.detach()
        stats["clip_frac"] += ((ratio - 1.0).abs() > clip).float().mean()
      n += 1
    self._guard(skipped, total_mb, "ppo")
    out = {k: float(v) / max(n, 1) for k, v in stats.items()}
    out["skipped_minibatches"] = float(skipped)
    return out

  def _guard(self, skipped: int, total: int, what: str) -> None:
    """Stop on persistent numerical failure; parameters must stay finite."""
    bad_params = [
      n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()
    ]
    if bad_params:
      raise RuntimeError(f"non-finite parameters after the {what} update: {bad_params}")
    key = f"_consec_{what}"
    streak = getattr(self, key, 0) + 1 if skipped else 0
    setattr(self, key, streak)
    if skipped:
      print(
        f"[FTSR-Ref] {what}: skipped {skipped}/{total} non-finite minibatches "
        f"(streak {streak})",
        flush=True,
      )
    if skipped > total // 2 or streak >= self.nan_max_consecutive:
      raise RuntimeError(
        f"persistent non-finite {what} updates ({skipped}/{total} minibatches, "
        f"{streak} consecutive iterations): stopping; last checkpoint is intact"
      )

  def _update_student(self) -> dict[str, float]:
    cfg = self.cfg
    total, n, count, skipped = 0.0, 0, 0, 0
    for mb in self.storage.minibatches(
      cfg["student_num_mini_batches"],
      cfg["student_num_learning_epochs"],
      self.teacher_mask,
    ):
      with torch.no_grad():
        target = self.model.teacher_encoder(mb["obs_teacher"])
      pred = self.model.student_encoder(mb["obs_policy"])
      loss = ((pred - target) ** 2).mean()
      self.student_optimizer.zero_grad()
      count += 1
      ok = bool(torch.isfinite(loss))
      if ok:
        loss.backward()
        g = nn.utils.clip_grad_norm_(
          self.model.student_encoder.parameters(), cfg["student_max_grad_norm"]
        )
        ok = bool(torch.isfinite(g))
      if not ok:
        self.student_optimizer.zero_grad()
        skipped += 1
        if skipped == 1:
          self._nonfinite_report(
            f"student_it{self.current_learning_iteration}",
            {"loss": loss, "pred": pred, "target": target, "obs": mb["obs_policy"]},
          )
        continue
      self.student_optimizer.step()
      total += loss.detach()
      n += 1
    self._guard(skipped, count, "student")
    return {"student_mse": float(total) / max(n, 1)}

  # ---------------------------------------------------------------------------
  # Periodic no-assist evaluation (separate process, does not block training).

  def _is_eval_iteration(self, it: int) -> bool:
    if not self.cfg.get("eval_task", ""):
      return False
    at = tuple(self.cfg.get("eval_at", ()) or ())
    every = self.cfg.get("eval_every", 0)
    if it in at:
      return True
    return bool(every) and it % every == 0 and (not at or it > max(at))

  def _spawn_eval(self, ckpt: str, it: int) -> None:
    out_dir = os.path.join(self.log_dir or ".", "eval")
    os.makedirs(out_dir, exist_ok=True)
    # Deterministic student, zero assistance, fixed seed and reset conditions
    # (analyze_recovery: recovery, h1 -> h2, foot-only, strict, time, qd / torque /
    # joint-limit statistics); also tracks the best checkpoint by autonomous recovery.
    cmd = [
      sys.executable,
      "-m",
      "minipi_getup.ftsr_ref.analyze_recovery",
      "periodic",
      "--checkpoint",
      ckpt,
      "--iteration",
      str(it),
      "--run-dir",
      self.log_dir or ".",
      "--per-pose",
      str(self.cfg.get("eval_envs_per_pose", 64)),
      "--task",
      self.cfg["eval_task"],
    ]
    log = open(os.path.join(out_dir, f"eval_{it}.log"), "w")
    self._eval_procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    print(f"[FTSR-Ref] spawned no-assist evaluation of {ckpt}", flush=True)

  # ---------------------------------------------------------------------------
  # Logging and I/O.

  def _log(self, it, logs, loss_info, adv_info, safety, rew_buf, len_buf, tc, tl):
    scalars: dict[str, float] = {k: sum(v) / len(v) for k, v in logs.items() if v}
    for k, v in loss_info.items():
      scalars["Loss/" + k] = v
    scalars["Loss/learning_rate"] = self.lr
    for k, v in adv_info.items():
      scalars["FTSR/" + k] = v
    for k, v in safety.items():
      scalars["Safety/" + k] = v
    scalars["Policy/mean_noise_std"] = float(self.model.std.mean())
    steps = self.cfg["num_steps_per_env"] * self.num_envs
    scalars["Perf/total_fps"] = steps / (tc + tl)
    if rew_buf:
      scalars["Train/mean_reward"] = statistics.mean(rew_buf)
      scalars["Train/mean_episode_length"] = statistics.mean(len_buf)
    if self.writer is not None:
      for k, v in scalars.items():
        self.writer.add_scalar(k, v, it)
    g = scalars.get
    print(
      f"[it {it}] rew {g('Train/mean_reward', float('nan')):.2f} "
      f"len {g('Train/mean_episode_length', float('nan')):.0f} "
      f"stage {g('Stage/stage', -1):.1f} S1 {g('Stage/frac_above_h1', 0):.2f} "
      f"S2 {g('Stage/frac_above_h2', 0):.2f} h_cmd {g('Stage/h_cmd', 0):.3f} F {g('Assist/force_mean', 0):.1f}N "
      f"tc {g('Assist/time_coeff', 0):.3f} | tau mean {safety.get('tau_mean', 0):.2f} "
      f"max {safety.get('tau_max', 0):.1f} | qd mean {safety.get('qd_mean', 0):.2f} "
      f"max {safety.get('qd_max', 0):.1f} >3 {safety.get('qd_frac_above_3', 0):.3f} >4 "
      f"{safety.get('qd_frac_above_4', 0):.3f} >6.28 "
      f"{safety.get('qd_frac_above_soft', 0):.4f} | mse "
      f"{loss_info.get('student_mse', 0):.4f} std "
      f"{scalars['Policy/mean_noise_std']:.2f} | inv "
      f"{loss_info.get('numerics_invalid_envs', 0):.0f} skip "
      f"{loss_info.get('skipped_minibatches', 0):.0f} ({tc:.1f}+{tl:.1f}s)",
      flush=True,
    )
    terms = self._reward_terms(scalars)
    if terms:
      half = (len(terms) + 1) // 2
      lines = (terms[:half], terms[half:]) if len(terms) > 6 else (terms,)
      for i, part in enumerate(lines):
        head = "  rewards: " if i == 0 else "           "
        print(head + " | ".join(part), flush=True)

  @staticmethod
  def _reward_terms(scalars: dict[str, float]) -> list[str]:
    """``name +x.xx`` for each logged reward term that is nonzero this iteration
    (zero-weight terms of the current stage are skipped)."""
    prefix = "Episode_Reward/"
    logged = {k[len(prefix) :]: v for k, v in scalars.items() if k.startswith(prefix)}
    order = [k for k in REWARD_SHORT if k in logged]
    order += sorted(k for k in logged if k not in REWARD_SHORT)
    return [
      f"{REWARD_SHORT.get(k, k)} {logged[k]:+.3f}"
      for k in order
      if abs(logged[k]) >= 5e-4
    ]

  def _report_transitions(self, it: int, logs: dict) -> None:
    stage = self.env.unwrapped.extras.get(STAGE_KEY)
    if stage is None or not stage.transitions:
      return
    g = {k: v[0] for k, v in logs.items() if v}
    for tr in stage.transitions:
      print(
        f"\n{'=' * 60}\n[FTSR STAGE TRANSITION] {tr['from']} -> {tr['to']}\n"
        f"  iteration = {it} (env step {tr['step']})\n"
        f"  S1 = {tr['s1']:.3f}  S2 = {tr['s2']:.3f}\n"
        f"  tc = {g.get('Assist/time_coeff', 0.0):.4f}  "
        f"assist_force (iteration mean) = {g.get('Assist/force_mean', 0.0):.1f} N\n"
        f"{'=' * 60}",
        flush=True,
      )
      if self.writer is not None:
        self.writer.add_scalar("Stage/transition_to", tr["to"], it)
    stage.transitions.clear()

  def save(self, path: str, infos=None) -> None:
    unwrapped = self.env.unwrapped
    stage = unwrapped.extras.get(STAGE_KEY)
    torch.save(
      {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "student_optimizer": self.student_optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "lr": self.lr,
        "dims": self.dims,
        "num_teacher": self.num_teacher,
        "rng": {
          "torch": torch.get_rng_state(),
          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "infos": {
          **(infos or {}),
          "env_state": {
            "common_step_counter": unwrapped.common_step_counter,
            **({"ftsr_stage": stage.state_dict()} if stage is not None else {}),
          },
        },
      },
      path,
    )

  def load_weights(self, path: str) -> None:
    """Initialize networks only (walking pretrain -> recovery). Iteration, env step
    counter (assist schedule) and optimizers start fresh."""
    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    assert ckpt["dims"] == self.dims, (ckpt["dims"], self.dims)
    self.model.load_state_dict(ckpt["model"])
    print(f"[FTSR-Ref] initialized weights from {path} (iter {ckpt['iter']})")

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
    """Resume everything."""
    del load_cfg
    ckpt = torch.load(
      path, map_location=map_location or self.device, weights_only=False
    )
    self.model.load_state_dict(ckpt["model"], strict=strict)
    if "optimizer" in ckpt:
      self.optimizer.load_state_dict(ckpt["optimizer"])
      self.student_optimizer.load_state_dict(ckpt["student_optimizer"])
    self.lr = ckpt.get("lr", self.lr)
    for group in self.optimizer.param_groups:
      group["lr"] = self.lr
    self.current_learning_iteration = ckpt.get("iter", 0)
    infos = ckpt.get("infos") or {}
    if "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
      stage = self.env.unwrapped.extras.get(STAGE_KEY)
      saved = infos["env_state"].get("ftsr_stage")
      if stage is not None and saved is not None:
        stage.load_state_dict(saved)
        print(f"[FTSR-Ref] restored stage {stage.stage} from {path}")
    rng = ckpt.get("rng")
    if rng is not None:
      torch.set_rng_state(rng["torch"].cpu())
      if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return infos

  def add_git_repo_to_log(self, *_args) -> None:
    if self.log_dir is None:
      return
    os.makedirs(self.log_dir, exist_ok=True)
    repo = os.path.dirname(os.path.abspath(__file__))
    out = []
    for args in (["rev-parse", "HEAD"], ["status", "--short"], ["diff", "HEAD"]):
      res = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=False
      )
      out.append(f"$ git {' '.join(args)}\n{res.stdout}")
    with open(os.path.join(self.log_dir, "git_state.txt"), "w") as f:
      f.write("\n".join(out))

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx") -> str:
    from minipi_getup.ftsr_ref.export import export_onnx

    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, filename)
    export_onnx(self.model, out)
    self.model.to(self.device)
    return out
