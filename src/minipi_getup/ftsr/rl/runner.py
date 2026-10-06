"""FTSR runner: force-guided, teacher-student PPO (paper Alg. 1) for mjlab.

Per iteration:

1. Rollout of ``num_steps_per_env`` steps. Env ``i < num_teacher`` belongs to the
   teacher group and acts on ``z = E^t(x_t)``; the others act on
   ``z = E^s(o_{t:t-H})`` (the "or" module). Both feed the shared actor
   ``pi(z, o_t)``. The critics always see ``E^t(x_t)``, since they only exist in
   training.
2. Constraint costs ``C_t = (|F|/mg, |T|/T_max) * dt`` of the Eq. 4 wrench applied
   during each step.
3. GAE for the reward and both costs, then Eq. 8:
   ``A_bar = std(A_r) - sum_i beta_i/(1-gamma) std(A_Ci)``.
4. PPO update of the actor, std, critic, cost critic and teacher encoder (paper
   Eq. 6: teacher rows backpropagate into E^t, student rows only into the actor).
   Unlike getup_gym's ``TSPPO.update``, teacher rows are found from a per-sample
   mask; the reference assumes they come first after shuffling.
5. Student encoder: MSE to the (updated) teacher latent over all samples (Eq. 10).

The stage-wise reward switching and the assistance live in the env (``ftsr.mdp``).
Plugs into mjlab's ``train`` / ``play`` scripts through
``register_mjlab_task(runner_cls=FtsrRunner)``.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import time
from collections import defaultdict, deque

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from minipi_getup.ftsr.mdp.assistance import COST_KEY
from minipi_getup.ftsr.rl.modules import FtsrModel, StudentPolicy
from minipi_getup.ftsr.rl.storage import FtsrStorage


def _scalar(v) -> float:
  if isinstance(v, torch.Tensor):
    return float(v.float().mean())
  return float(v)


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
    self.groups = ("actor", "student", "teacher", "critic")
    dims = {g: obs[g].shape[-1] for g in self.groups}
    self.dims = dims
    self.num_envs = env.num_envs
    self.num_actions = env.num_actions
    self.num_teacher = int(round(train_cfg["teacher_fraction"] * self.num_envs))
    self.teacher_mask = torch.arange(self.num_envs, device=device) < self.num_teacher

    self.model = FtsrModel(
      actor_obs_dim=dims["actor"],
      student_obs_dim=dims["student"],
      teacher_obs_dim=dims["teacher"],
      critic_obs_dim=dims["critic"],
      num_actions=self.num_actions,
      latent_dim=train_cfg["latent_dim"],
      actor_hidden=tuple(train_cfg["actor_hidden_dims"]),
      critic_hidden=tuple(train_cfg["critic_hidden_dims"]),
      encoder_hidden=tuple(train_cfg["encoder_hidden_dims"]),
      activation=train_cfg["activation"],
      init_noise_std=train_cfg["init_noise_std"],
    ).to(device)
    self.lr = train_cfg["learning_rate"]
    self.optimizer = torch.optim.Adam(self.model.ppo_parameters(), lr=self.lr)
    self.student_optimizer = torch.optim.Adam(
      self.model.student_encoder.parameters(), lr=train_cfg["student_learning_rate"]
    )
    self.storage = FtsrStorage(
      train_cfg["num_steps_per_env"],
      self.num_envs,
      dims,
      self.num_actions,
      train_cfg["latent_dim"],
      num_costs=2,
      device=device,
    )
    print(
      f"[FTSR] envs={self.num_envs} teacher={self.num_teacher} "
      f"student={self.num_envs - self.num_teacher} obs dims={dims}"
    )
    print(self.model)

    init = train_cfg.get("init_checkpoint") or ""
    if init:
      self.load_weights(init)

  # ---------------------------------------------------------------------------
  # Acting.

  def _latents(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_t = self.model.teacher_encoder(obs["teacher"])
    z_s = self.model.student_encoder(obs["student"])
    z = torch.where(self.teacher_mask.unsqueeze(-1), z_t, z_s)
    return z_t, z_s, z

  def _cost(self) -> torch.Tensor:
    cost = self.env.unwrapped.extras.get(COST_KEY)
    if cost is None:
      return torch.zeros(self.num_envs, 2, device=self.device)
    return cost.clone()

  def get_inference_policy(self, device=None, mode: str = "student"):
    """Deterministic policy on a TensorDict of observation groups.

    ``mode="student"`` is the deployable one (proprioceptive history only).
    """
    self.model.eval()
    if device is not None:
      self.model.to(device)
    model = self.model

    def policy(obs):
      with torch.no_grad():
        if mode == "teacher":
          z = model.teacher_encoder(obs["teacher"])
        else:
          z = model.student_encoder(obs["student"])
        return model.actor_mean(z, obs["actor"])

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
      # Desynchronize the first episodes, but only over their first quarter (5 s):
      # envs started near 10 s would hit the "on the ground for 10 s" termination
      # right away.
      unwrapped.episode_length_buf = torch.randint_like(
        unwrapped.episode_length_buf, high=int(unwrapped.max_episode_length) // 4
      )
    monitor = unwrapped.action_manager.get_term("joint_pos").monitor
    action_mgr = unwrapped.action_manager

    obs = self.env.get_observations().to(self.device)
    rew_buf: deque[float] = deque(maxlen=200)
    len_buf: deque[float] = deque(maxlen=200)
    cur_rew = torch.zeros(self.num_envs, device=self.device)
    cur_len = torch.zeros(self.num_envs, device=self.device)
    gamma, lam = cfg["gamma"], cfg["lam"]
    start_it = self.current_learning_iteration
    end_it = start_it + num_learning_iterations

    for it in range(start_it, end_it):
      t0 = time.time()
      self.model.train()
      logs: dict[str, list[float]] = defaultdict(list)
      motion = defaultdict(float)
      monitor.clear()
      with torch.inference_mode():
        for _ in range(cfg["num_steps_per_env"]):
          z_t, z_s, z = self._latents(obs)
          mean = self.model.actor_mean(z, obs["actor"])
          std = self.model.std.expand_as(mean)
          dist = torch.distributions.Normal(mean, std)
          actions = dist.sample()
          value = self.model.value(z_t, obs["critic"])
          cost_value = self.model.cost_value(z_t, obs["critic"])
          cost = self._cost()
          step_obs = {g: obs[g] for g in self.groups}

          obs, rewards, dones, extras = self.env.step(actions)
          obs = obs.to(self.device)
          dones_f = dones.float()
          rewards = rewards.clone()
          if "time_outs" in extras:
            to = extras["time_outs"].float()
            rewards += gamma * value.squeeze(-1) * to
            cost = cost + gamma * cost_value * to.unsqueeze(-1)
          self.storage.add(
            obs=step_obs,
            z_student=z_s,
            actions=actions,
            log_prob=dist.log_prob(actions).sum(-1),
            mu=mean,
            sigma=std,
            rewards=rewards,
            costs=cost,
            dones=dones_f,
            values=value.squeeze(-1),
            cost_values=cost_value,
          )

          for k, v in extras.get("log", {}).items():
            logs[k].append(_scalar(v))
          a, a1, a2 = (
            action_mgr.action,
            action_mgr.prev_action,
            action_mgr.prev_prev_action,
          )
          motion["action_rate"] += float(torch.sum((a - a1) ** 2, -1).mean())
          motion["action_smoothness"] += float(
            torch.sum((a - 2 * a1 + a2) ** 2, -1).mean()
          )
          robot = unwrapped.scene["robot"]
          motion["joint_vel_abs"] += float(robot.data.joint_vel.abs().mean())

          cur_rew += rewards
          cur_len += 1
          done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
          if len(done_ids) > 0:
            rew_buf.extend(cur_rew[done_ids].tolist())
            len_buf.extend(cur_len[done_ids].tolist())
            cur_rew[done_ids] = 0.0
            cur_len[done_ids] = 0.0

      # Outside inference mode: returns/advantages must be ordinary tensors, since the
      # PPO loss saves them for backward.
      with torch.no_grad():
        z_t = self.model.teacher_encoder(obs["teacher"])
        last_value = self.model.value(z_t, obs["critic"])
        last_cost_value = self.model.cost_value(z_t, obs["critic"])
        adv_info = self.storage.compute_advantages(
          last_value,
          last_cost_value,
          gamma,
          lam,
          tuple(cfg["penalty_factors"]),
          cfg["use_force_guidance"],
        )
      t_collect = time.time() - t0

      t1 = time.time()
      loss_info = self._update()
      loss_info.update(self._update_student())
      self.storage.clear()
      t_learn = time.time() - t1
      self.current_learning_iteration = it + 1

      steps = cfg["num_steps_per_env"]
      safety = monitor.summary()
      safety.update({k: v / steps for k, v in motion.items()})
      self._log(
        it, logs, loss_info, adv_info, safety, rew_buf, len_buf, t_collect, t_learn
      )
      if self.log_dir is not None and (
        (it + 1) % cfg["save_interval"] == 0 or it == end_it - 1
      ):
        self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
    if self.writer is not None:
      self.writer.flush()

  def _update(self) -> dict[str, float]:
    cfg = self.cfg
    clip = cfg["clip_param"]
    stats = defaultdict(float)
    n = 0
    for mb in self.storage.minibatches(
      cfg["num_mini_batches"], cfg["num_learning_epochs"], self.teacher_mask
    ):
      z_t = self.model.teacher_encoder(mb["obs_teacher"])
      # Teacher rows: fresh E^t latent (gradient into E^t). Student rows: the latent
      # used in the rollout, detached (Eq. 6, j = s updates theta only).
      z = torch.where(mb["teacher"].unsqueeze(-1), z_t, mb["z_student"])
      mean = self.model.actor_mean(z, mb["obs_actor"])
      std = self.model.std.expand_as(mean)
      dist = torch.distributions.Normal(mean, std)
      log_prob = dist.log_prob(mb["actions"]).sum(-1)
      entropy = dist.entropy().sum(-1).mean()
      value = self.model.value(z_t, mb["obs_critic"]).squeeze(-1)
      cost_value = self.model.cost_value(z_t.detach(), mb["obs_critic"])

      if cfg["schedule"] == "adaptive" and cfg["desired_kl"] > 0:
        with torch.no_grad():
          old_mu, old_sigma = mb["mu"], mb["sigma"]
          kl = torch.sum(
            torch.log(std / old_sigma + 1e-5)
            + (old_sigma**2 + (old_mu - mean) ** 2) / (2.0 * std**2)
            - 0.5,
            dim=-1,
          ).mean()
          if kl > cfg["desired_kl"] * 2.0:
            self.lr = max(1e-5, self.lr / 1.5)
          elif 0.0 < kl < cfg["desired_kl"] / 2.0:
            self.lr = min(1e-2, self.lr * 1.5)
          for group in self.optimizer.param_groups:
            group["lr"] = self.lr
          stats["kl"] += float(kl)

      adv = mb["advantages"]
      ratio = torch.exp(log_prob - mb["log_prob"])
      surrogate = -torch.min(
        adv * ratio, adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
      ).mean()
      if cfg["use_clipped_value_loss"]:
        v_clip = mb["values"] + (value - mb["values"]).clamp(-clip, clip)
        value_loss = torch.max(
          (value - mb["returns"]) ** 2, (v_clip - mb["returns"]) ** 2
        ).mean()
      else:
        value_loss = ((value - mb["returns"]) ** 2).mean()
      cost_value_loss = ((cost_value - mb["cost_returns"]) ** 2).mean()

      loss = (
        surrogate
        + cfg["value_loss_coef"] * (value_loss + cost_value_loss)
        - cfg["entropy_coef"] * entropy
      )
      self.optimizer.zero_grad()
      loss.backward()
      grad_norm = nn.utils.clip_grad_norm_(
        self.model.ppo_parameters(), cfg["max_grad_norm"]
      )
      self.optimizer.step()
      with torch.no_grad():
        self.model.std.clamp_(min=cfg["min_noise_std"])

      stats["surrogate"] += float(surrogate)
      stats["value"] += float(value_loss)
      stats["cost_value"] += float(cost_value_loss)
      stats["entropy"] += float(entropy)
      stats["grad_norm"] += float(grad_norm)
      stats["clip_frac"] += float(((ratio - 1.0).abs() > clip).float().mean())
      n += 1
    return {k: v / n for k, v in stats.items()}

  def _update_student(self) -> dict[str, float]:
    """Student encoder MSE to the teacher latent over all samples (paper Eq. 10)."""
    cfg = self.cfg
    total, n = 0.0, 0
    for mb in self.storage.minibatches(
      cfg["student_num_mini_batches"],
      cfg["student_num_learning_epochs"],
      self.teacher_mask,
    ):
      with torch.no_grad():
        target = self.model.teacher_encoder(mb["obs_teacher"])
      pred = self.model.student_encoder(mb["obs_student"])
      loss = ((pred - target) ** 2).mean()
      self.student_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(
        self.model.student_encoder.parameters(), cfg["student_max_grad_norm"]
      )
      self.student_optimizer.step()
      total += float(loss)
      n += 1
    return {"student_mse": total / n}

  # ---------------------------------------------------------------------------
  # Logging and I/O.

  def _log(self, it, logs, loss_info, adv_info, safety, rew_buf, len_buf, tc, tl):
    steps = self.cfg["num_steps_per_env"] * self.num_envs
    w = self.writer
    scalars: dict[str, float] = {}
    for k, vals in logs.items():
      scalars[k] = sum(vals) / len(vals)
    for k, v in loss_info.items():
      scalars["Loss/" + k] = v
    scalars["Loss/learning_rate"] = self.lr
    for k, v in adv_info.items():
      scalars["FTSR/" + k] = v
    for k, v in safety.items():
      scalars["Safety/" + k] = v
    scalars["Policy/mean_noise_std"] = float(self.model.std.mean())
    scalars["Perf/total_fps"] = steps / (tc + tl)
    scalars["Perf/collection_time"] = tc
    scalars["Perf/learning_time"] = tl
    if len(rew_buf) > 0:
      scalars["Train/mean_reward"] = statistics.mean(rew_buf)
      scalars["Train/mean_episode_length"] = statistics.mean(len_buf)
    if w is not None:
      for k, v in scalars.items():
        w.add_scalar(k, v, it)
    print(
      f"[FTSR it {it}] rew {scalars.get('Train/mean_reward', float('nan')):.2f} "
      f"len {scalars.get('Train/mean_episode_length', float('nan')):.0f} "
      f"stage {scalars.get('Stage/stage', -1):.0f} "
      f"S1 {scalars.get('Stage/frac_above_h1', 0):.2f} "
      f"S2 {scalars.get('Stage/frac_above_h2', 0):.2f} "
      f"F {scalars.get('Assist/force_mean', 0):.1f}N "
      f"stood {scalars.get('Episode_Metrics/stood_up', float('nan')):.2f} "
      f"tau p99 {safety.get('torque_p99', 0):.1f} max {safety.get('torque_max', 0):.1f} "
      f"sat {safety.get('torque_sat_frac', 0):.4f} "
      f"mse {loss_info.get('student_mse', 0):.4f} "
      f"std {scalars['Policy/mean_noise_std']:.2f} "
      f"({tc:.2f}+{tl:.2f}s)",
      flush=True,
    )

  def save(self, path: str, infos=None) -> None:
    torch.save(
      {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "student_optimizer": self.student_optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "lr": self.lr,
        "dims": self.dims,
        "infos": {
          **(infos or {}),
          "env_state": {"common_step_counter": self.env.unwrapped.common_step_counter},
        },
      },
      path,
    )

  def load_weights(self, path: str) -> None:
    """Initialize the networks from a checkpoint without resuming its run.

    Used to start FTSR from the r_w pretraining (paper Sec. III-A). The iteration
    counter, env step counter (assist schedule) and optimizer state start fresh.
    """
    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    self.model.load_state_dict(ckpt["model"])
    print(f"[FTSR] initialized weights from {path} (iter {ckpt.get('iter')})")

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
    """Resume: weights, optimizers, iteration and env step counter."""
    del load_cfg
    ckpt = torch.load(
      path, map_location=map_location or self.device, weights_only=False
    )
    self.model.load_state_dict(ckpt["model"], strict=strict)
    if "optimizer" in ckpt:
      self.optimizer.load_state_dict(ckpt["optimizer"])
      self.student_optimizer.load_state_dict(ckpt["student_optimizer"])
    self.lr = ckpt.get("lr", self.lr)
    self.current_learning_iteration = ckpt.get("iter", 0)
    infos = ckpt.get("infos") or {}
    if "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def add_git_repo_to_log(self, *_args) -> None:
    """Record the source state (commit and diff) next to the run's checkpoints."""
    if self.log_dir is None:
      return
    os.makedirs(self.log_dir, exist_ok=True)
    repo = os.path.dirname(os.path.abspath(__file__))
    out = []
    for args in (["rev-parse", "HEAD"], ["status", "--short"], ["diff", "HEAD"]):
      try:
        res = subprocess.run(
          ["git", "-C", repo, *args], capture_output=True, text=True, check=False
        )
        out.append(f"$ git {' '.join(args)}\n{res.stdout}")
      except OSError as e:
        out.append(f"$ git {' '.join(args)}\nfailed: {e}")
    with open(os.path.join(self.log_dir, "git_state.txt"), "w") as f:
      f.write("\n".join(out))

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx") -> str:
    """Export the student (deployable) policy: (o_{t:t-H}, o_t) -> a_t."""
    policy = StudentPolicy(self.model).to("cpu").eval()
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, filename)
    dummy = (torch.zeros(1, self.dims["student"]), torch.zeros(1, self.dims["actor"]))
    torch.onnx.export(
      policy,
      dummy,
      out,
      input_names=["student_obs_history", "actor_obs"],
      output_names=["actions"],
      opset_version=18,
      dynamo=False,
    )
    self.model.to(self.device)
    return out
