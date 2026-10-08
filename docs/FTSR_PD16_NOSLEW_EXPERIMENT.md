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
