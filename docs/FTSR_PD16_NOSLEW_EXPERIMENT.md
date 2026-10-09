# Experiment pd16_noslew: plain 16 Nm PD plant, no target slew (FTSR v2 method)

Branch `ftsr-only`, worktree `/home/huy/minipi_getup_ftsr` (the main checkout
`/home/huy/minipi_getup` stays on its own branch). Started 2026-10-08. Previous
experiments (walk v1, recovery v1/v2, v3 prepared) are documented in
`FTSR_REFERENCE_EXPERIMENT_LOG.md` and are not modified.

## What changed (user request 2026-10-08)

| Item | v2 | pd16_noslew |
|---|---|---|
| Motor model | four-quadrant torque-speed envelope (H-conservative: omega0 7.85 rad/s, stall 21 Nm, cap 16) written per physics step | `tau = clip(kp (q* - q) - kd qdot, -16, 16)`: MuJoCo position actuator, static `forcerange` +-16 Nm; no velocity derating; H-conservative / H-loose and `motor_envelope` removed |
| Target slew | 0.06 rad per 20 ms step on q* | none: `raw -> clip +-50 -> scale -> XML range clip -> PD` |
| `qd_soft_envelope` | `sum relu(|qd| - 3)^2`, -0.01 all stages | `sum relu(|qd| - 6.28)^2`, -0.01 all stages |
| Passive window | kp 0 / kd 1, 2 s | unchanged (gains written once per policy step) |
| Monitoring | slew saturation | fraction of samples with |qd| > 6.28 (training line `>6.28`, evaluations) |
| PPO update | no guard | non-finite guard (below) |
| Periodic evaluation | `evaluate.py` every 500 | `analyze_recovery periodic` at 500, 1000, 1500, 2000, 2500, 2750, 3000, 3500, then every 500; best checkpoint by autonomous recovery |

Unchanged (FTSR v2 method, checked by validate test 28): h1/h2/h3 = 0.190 / 0.276 /
0.345 m; height-reward targets 0.214 / 0.276 / 0.345 m; stateless stages that may
regress (task `Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless`), 2/3 rule; Eq. 4 assistance
(F_max 1.474 m g, decay to iteration 3000); Eq. 5-8 (beta 0.001); every reward term
and weight except the qd envelope limit; PPO, entropy 0.01, initial std 1.0,
teacher/student networks; four fallen poses, 2 s passive window; physics 0.5 ms x 40
(50 Hz policy); observations (48 / 240 / 20 / 50), action dimension 12, action
scales, raw clip 50. The monotonic stage (v3) is not active.

Other velocity-dependent terms (unchanged, reported separately as asked):
`pen_dof_vel_l2` (sum qd^2, -4e-3, stage r_w only), `pen_dof_acc_l2` (sum of squared
qd differences / 0.02 s, all stages), `pen_joint_power_l2` (sum |qd||tau| below
0.322 m, -1e-4, r_w only). `pen_max_velocity_l2` uses the base velocity, not joints;
`pen_action_rate_l2` / `pen_action_smoothness_l2` act on actions.

## Numerical guard (does not change the objective while values are finite)

`FtsrRunner._update` / `_update_student`: a minibatch whose loss or clipped gradient
norm is non-finite is skipped (no optimizer step); its inputs (loss, ratio,
advantages, returns, values, observations, ...) are written once per iteration to
`RUN/nonfinite_<update>_it<N>.json`. Training stops with an error (last checkpoint
intact) if more than half of an iteration's minibatches are skipped, if skips occur in
3 consecutive iterations, or if any parameter becomes non-finite. Skips are logged
(`Loss/skipped_minibatches`). Motivation: v2 crashed at iteration 2279 with NaN actor
parameters after a PPO update.

## Validation (`python -m minipi_getup.ftsr_ref.validate`): 29/29 pass (123 s)

| # | Check | Result |
|---|---|---|
| 7 | target = clip(scale * clip(a, +-50)), inside XML ranges; no slew | error 1.2e-7; consecutive targets jump up to 3.0 rad |
| 9 | no torque-speed derating | floating hip pitch / calf: saturated motoring at |qd| up to 35.6 / 33.7 rad/s with |force| = 16 Nm (error 0) |
| 10 | force = clip(kp (q*-q) - kd qd, +-16) every physics step (incl. passive kp 0 / kd 1) | max error 1.7e-5 (walk), 1.9e-5 (recovery); |tau| max 16.00 |
| 11 | qd penalty | 0 at |qd| = 6.28, 6.0; 1.0 at 7.28; weight -0.01 all stages |
| 12-14, 16, 25 | observation layout, history, last action, teacher/student split, ONNX [1,240] -> [1,12] | unchanged, pass |
| 23 | random rollouts (action std 2 and 50), walk and recovery | 0 non-finite |
| 28 | training task keeps the v2 method | pass |
| 29 | all-NaN advantages -> every minibatch skipped, parameters unchanged, training stops | pass |

Smoke run (64 envs, 3 iterations, random policy): terminal shows the reward breakdown,
h_cmd and qd > 6.28 (36 % of samples with a random policy: the plant now allows ~36
rad/s); the periodic evaluation writes `eval/eval_N.json` and `best_recovery*.`

## Runs

Pipeline `tools/ftsr_pd16_pipeline.sh` (tmux session `ftsr_pd16`):

1. Walk from scratch: `uv run train Mjlab-FTSR-Ref-MiniPi-Walk --env.scene.num-envs 4000
   --agent.max-iterations 900 --agent.run-name ftsr_walk_pd16_noslew`
   (log `logs/ftsr_walk_pd16_noslew.log`).
2. `evaluate.py` on its `model_900.pt` -> `eval_walk_900.json`. Acceptance: vx tracking
   correlation >= 0.9 and fall rate <= 10 % (physical metrics reported, not gating).
   Rejected -> the pipeline stops before recovery.
3. Recovery: `uv run train Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless
   --env.scene.num-envs 4000 --agent.init-checkpoint <walk model_900.pt>
   --agent.run-name ftsr_recovery_pd16_noslew` (8000 iterations, assistance to 3000;
   log `logs/ftsr_recovery_pd16_noslew.log`; evaluations in `RUN/eval/eval_N.json`,
   best in `RUN/best_recovery.json` / `best_recovery_model.pt`).

Nothing here is hardware-verified; no checkpoint is labelled hardware-ready.

## Results

(appended during the run)

### Walk `ftsr_walk_pd16_noslew` (2026-10-08 22:59-23:50, 900 iterations, 3.2 s/it)

Run `logs/rsl_rl/minipi_ftsr_ref_walk/2026-10-08_22-59-37_ftsr_walk_pd16_noslew`. No
non-finite updates, KL 0.010-0.012, grad norm 3.6-7.7, std 1.0 -> 0.07 (converged), no
falls in training after it ~150. Final rewards /s: height 4.90, lin_vel 6.80, yaw 3.16
(still rising), qd > 6.28 rad/s 0.3 % of training samples.

`eval_walk_900.json` (deterministic student, 16 envs per command, 10 s): **ACCEPTED**
(corr_vx 1.000, fall rate 0 %).

- vx: -0.34 -> -0.32, -0.15 -> -0.15, 0.15 -> 0.14, 0.30 -> 0.29, 0.54 -> 0.52 m/s.
  Zero command: drift 3.6 cm, 0.14 touchdowns/s (stands still).
- Turning while walking (vx 0.2, wz +-0.44): -0.54 / +0.45 rad/s (v0 walk: +-0.38).
- Turning in place (vx 0, wz +-0.44): still none (-0.03 / 0.00 rad/s), as in v0.
- Physical: qd p99 per joint 2.1-7.1 rad/s (left calf 7.1, ankle pitch 5.7-5.8), qd >
  6.28 0.19 % of samples, torque max 16 Nm, near cap 0.08 %, > 6 Nm 6.6 % (v0 1.5 %),
  joint-limit overshoot max 0.026 rad.

### Recovery `ftsr_recovery_pd16_noslew` started 23:51 from that model_900

Note: the walking init ends with std 0.07 (v0 walk init: 0.31 at recovery start), so
recovery starts with little exploration; the entropy bonus (0.01, unchanged) is the
only mechanism that raises it. Watched, not changed.

Periodic no-assist evaluations (deterministic student, 64 envs per pose, seed 2150;
recovered = last 3 s base > h1 and tilt < 18 deg):

| it | recovered mean | supine | prone | left | right |
|---|---|---|---|---|---|
| 500, 1000, 1500, 2000 | 0 % | 0 | 0 | 0 | 0 |
| 2500 | 41.4 % | 52 % | 23 % | 48 % | 42 % |

`best_recovery_model.pt` = model_2500 (v2 model_2150: about 13 %).

### Stop at iteration 2700: non-finite constraint cost (physics explosion)

Training stopped at iteration 2700 (tc 0.10) with `persistent non-finite ppo updates
(20/20 minibatches)`. The guard worked as designed; `model_2700.pt` and all parameters
are finite. Dump `nonfinite_ppo_it2700.json`: rewards, returns, values, observations,
log-probs and ratios finite; 5 non-finite advantages in the dumped minibatch.

Cause: one env's physics state became non-finite within a substep (joint speeds of
44-48 rad/s occur in this plant). mjlab handled the env itself (`nan` termination
reset it, the reward manager zeroed its non-finite rewards), but the Eq. 4 constraint
cost accumulated in `FtsrJointAction.apply_actions` from that substep's base height /
orientation was NaN. `discounted_cost_to_go` propagated it backwards through the
episode, so `A_bar` (Eq. 8) and every PPO loss were NaN. The v2 crash at iteration
2279 (NaN actor parameters, before the guard existed) very likely had the same cause.

Fix (commit 9a6796a; the objective is unchanged on finite data):

1. Source: an env with non-finite q, qd, actuator force or Eq. 4 wrench in any substep
   is flagged invalid for the policy step; its wrench and costs are zero; it is kept
   out of the substep monitor.
2. Storage (second guard): a non-finite reward, value or cost also makes the sample
   invalid. Invalid samples truncate the trajectory at the start of their step
   (advantage 0; the previous step bootstraps V(s_t) of the finite pre-step
   observation), carry no cost and are excluded from advantage standardization.
3. PPO: surrogate, value loss, entropy and KL are averaged over valid samples only
   when an invalid sample exists; otherwise the original code runs. Invalid episodes
   are left out of the reward statistics. The env itself is reset by mjlab's `nan`
   termination; an explosion is never a successful or valid transition.
4. Logging: every explosion is printed (`PHYSICS EXPLOSION`, env ids, pre-explosion
   max |qd| and base height) and appended to `RUN/explosions.jsonl`; TensorBoard
   `Loss/numerics_*`, `FTSR/invalid_samples`, `FTSR/nonfinite_cost_samples`,
   `Loss/skipped_minibatches`; terminal `inv` (envs) and `skip` (minibatches).
5. Stop rule: more than 8 envs exploding in one iteration, or explosions in more than
   5 of the last 100 iterations, raises with a report (last checkpoint intact).

Validation 32/32 (tests 30-32 new): poisoned env flagged, zero cost, nan-terminated
and reset, valid in the next step, monitor finite; storage returns / advantages on
all-valid rollouts bitwise equal to the pre-fix code (36c1bc6), with and without
Eq. 8; a NaN sample gives finite advantages, 0 at that sample, and results
independent of its contents; the PPO update on an all-valid rollout is bitwise equal
to the pre-fix `_update`, and with an invalid sample it is finite and independent of
that sample. (Two separate GPU env instances differ by ~1e-3 even without poisoning,
so test 30 checks the sanitation identity on the clean tensors instead of a twin
comparison.)

### Resume from model_2700 (2026-10-09 08:33)

`tools/ftsr_pd16_resume.sh`: `--agent.resume True --agent.load-run
2026-10-08_23-51-06_ftsr_recovery_pd16_noslew --agent.load-checkpoint model_2700.pt
--agent.max-iterations 5300` (to 8000). Restored: networks, both Adam states, adaptive
lr, iteration 2700, env step counter (tc 0.094 at it 2718, continuing the schedule),
RNG states; the stage is recomputed (stateless). Not restorable: the physics state of
the 4000 envs (fresh resets with random episode phases), so the first reported
episode rewards after resume are from short episodes. New run dir
`logs/rsl_rl/minipi_ftsr_ref/2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700`
(log `logs/ftsr_recovery_pd16_noslew_resume2700.log`), which starts with the original
best record (model_2500) and evaluations; evaluations at 2750, 3000, 3500, then every
500. Unchanged: plant, torque cap, qd penalty, rewards, stage thresholds, PPO, assist
schedule.
