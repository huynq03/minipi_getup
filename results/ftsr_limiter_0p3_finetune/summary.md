# Limiter fine-tune (0.3 rad/policy step) from model_4000

This is the adaptation of an FTSR-trained recovery policy to a PD-target rate limiter. It is
simulation only and **not hardware-verified**. The limiter bounds the **PD-target** change
to 0.3 rad per 20 ms, about 15 rad/s of target slew. It is **not a hard limit on joint
speed**: the joints still reach 21–32 rad/s through PD error, contacts and inertial coupling.

## What was done

- **Code** (commit 48b89cc; tests 33 and 34 pass, as do the relevant earlier tests 6, 7, 8, 10, 14, 22, 24, 28 and 30):
  - `FtsrActionCfg.target_rate_limit` defaults to 0, which disables it, so the baseline is unchanged.
  - When it is on, the limiter is `q* = q*_prev + clip(clip(q_cmd, q_min, q_max) − q*_prev, ±0.3)`.
    `q*_prev` is the clipped measured pose in the passive window and at the first actuated
    step of every episode, per env.
  - The training implementation shares `rate_limit_target()` with the evaluation wrapper.
    Test 33 shows both are bit-exact to the reference at every step, including across a
    forced mid-run reset. It also checks that the raw action and `last_action` are unchanged.
  - New task: `Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3`. It is the stateless task
    plus the limiter in both the train and play cfg. Test 34 verifies that it is otherwise
    identical, and that the evaluation refuses to apply the limiter twice.
- **Fine-tune** (`tools/ftsr_limiter_finetune.sh`):
  - Run `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000`, log `logs/ftsr_limit0p3_ft4000.log`.
  - Source: `2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700/model_4000.pt`, sha256 `408df87e…0ad3`.
    Provenance is in `finetune_source.json` in the run dir.
  - Restored: networks (actor, critic, teacher and student encoders, std), both Adam states,
    the adaptive lr, the iteration counter (4000) and `common_step_counter` (96 000, so
    t > t_tag).
  - 1000 **additional** iterations (4000 → 5000), 4000 envs.
  - Unchanged: rewards, entropy, the lr schedule (adaptive, KL-driven), std, PPO, the
    teacher/student split, force guidance, observations, commands, poses, the passive window,
    PD gains, the 16 Nm cap, ranges and physics.
- **Stability of the training run:**
  - Assistance stayed at zero throughout: `Assist/time_coeff` = 0 and force max = 0 at every iteration.
  - 0 invalid envs, 0 explosions, 0 skipped minibatches.
  - The adaptive lr moved within its normal range, from 1.5e-5 to about 1e-4. That is the
    unchanged KL schedule.

## Evaluation protocol

- Deterministic student, zero assistance, play cfg (no observation noise).
- The limiter comes from the task cfg and is applied once.
- Verification at every step in all 15 runs: maximum |Δq*| = 0.3000001; 0 targets outside the
  range; the base at entry equals the measured pose exactly (512/512); raw-action mismatch is 0.
- Seeds 2150, 2151 and 2152, 128 episodes per pose for each.
- The initial states are identical for every checkpoint of a seed: maximum |Δqpos₀| = 0.
- **Baseline: model_4000 with the same limiter, zero-shot.**
- Fine-tuned checkpoints evaluated: 4250, 4500, 4750 and 5000, i.e. every 250 additional iterations.

## Result (3 seeds pooled, 1536 episodes per row)

| | baseline model_4000 + 0.3 | ft 4250 | **ft 4500 (selected)** | ft 4750 | ft 5000 |
|---|---|---|---|---|---|
| Recovery | 98.3 % | 99.9 % | **100 %** | 99.9 % | 100 % |
| Worst pose / seed | 96.1 % | 99.2 % | **100 %** | 99.2 % | 100 % |
| Strict > 0.31 m, overall (worst pose / seed) | 97.9 (95.3) | 99.8 (99.2) | **100 (100)** | 99.9 (99.2) | 99.9 (99.2) |
| Foot-only stable, overall (worst pose / seed) | 98.2 (95.3) | 99.9 (99.2) | **100 (100)** | 99.9 (99.2) | 100 (100) |
| Failures out of 1536 | 26 | 2 | **0** | 1 | 0 |
| Recovery time, median / p95 (s) | 1.32 / 8.04 | 0.56 / 1.53 | **0.48 / 1.03** | 0.50 / 0.86 | 0.50 / 0.87 |
| Events > 20 rad/s per episode | 0.85 | 0.30 | **0.19** | 0.26 | 0.29 |
| Events > 20 in the first 100 ms (count) | 74 | 158 | 146 | 165 | 190 |
| Phase A qd p99 / max (rad/s) | 15.1 / 28.7 | 15.5 / 26.7 | 15.6 / 25.4 | 15.7 / 25.7 | 15.7 / 25.2 |
| Phase B qd p99 / max | 15.4 / 32.8 | 15.8 / 22.3 | 15.8 / 24.5 | 15.6 / 31.7 | 15.7 / 23.5 |
| Phase C qd p99 / max | 15.0 / 33.5 | 14.3 / 25.1 | 14.2 / 26.7 | 13.6 / 22.8 | 14.1 / 23.3 |
| Phase D qd p99 / max | 5.8 / 28.8 | 5.4 / 29.1 | 6.4 / 21.5 | 5.7 / 24.2 | 6.1 / 17.7 |
| Overall qd max | 33.5 | 29.1 | 26.6 | 31.7 | **25.2** |
| Joint-limit penetration max (rad) | 0.381 | 0.327 | 0.241 | 0.278 | **0.155** |
| Episodes with penetration > 0.05 rad | 86.5 % | 65.8 % | 63.3 % | 67.4 % | **61.4 %** |
| Torque peak / saturation fraction | 16 Nm / 0.58 % | 16 / 0.29 % | 16 / 0.46 % | 16 / 0.55 % | 16 / 0.58 % |
| Maximum tracking error (rad) | 1.97 | 2.23 | 1.75 | 1.90 | **1.63** |
| Root constraint force max (body weights) | 17.6 | 16.9 | 16.0 | 17.4 | 16.1 |
| Invalid / NaN episodes | 0 | 0 | 0 | 0 | 0 |
| Limiter active (share of joint-steps clipped) | 5.5 % | 4.7 % | 8.5 % | 9.9 % | 13.4 % |
| Policy std (training) | 1.75 | 1.84 | 1.95 | 2.06 | 2.16 |
| Student MSE (training) | 0.95 (at 4010) | 0.64 | 0.67 | 0.74 | 0.78 |

Per seed and per pose, as recovered / strict / foot-only (%). The full table is in `per_pose_results.csv`.

| | Seed 2150 S / P / L / R | Seed 2151 | Seed 2152 |
|---|---|---|---|
| baseline 4000 | 100 / 97.7 / 97.7 / 97.7 | 99.2 / 98.4 / 99.2 / 97.7 | 99.2 / **96.1** / 99.2 / 97.7 |
| **ft 4500** | 100 / 100 / 100 / 100 (all three metrics) | 100 for all | 100 for all |
| ft 5000 | 100 for all | supine strict 99.2, otherwise 100 | left strict 99.2, otherwise 100 |

Failure categories of the baseline across 3 seeds: 11 high-but-tilted, 8 reached h1 but not
held, 3 fell after recovery, 2 partial lift, 2 never lifted. The fine-tuned 4500 has none.

## Selection: model_4500

The selection criteria, in order:

1. **≥ 95 % on every pose of every seed.** All fine-tuned checkpoints pass. 4500 and 5000
   reach 100 %, and **only 4500 is 100 % on strict standing and foot-only stability in every
   pose and seed**. These are the preferred criteria.
2. **Spike events and penetration no worse than the baseline.** 4500 is better on both:
   - Events: 0.85 → 0.19 per episode, −78 %, the lowest of all checkpoints.
   - Penetration maximum: 0.381 → 0.241, −37 %.
   - Episodes with penetration > 0.05 rad: 86.5 → 63.3 %.
3. **Recovery time is not forced down.** 4500 is faster than the baseline (median 0.48 s
   instead of 1.32 s, p95 1.03 s instead of 8.04 s) because the stalls disappeared.

Against model_5000:

- **5000 is better on penetration** (max 0.155, > 0.05 in 61.4 % of episodes) and on peak
  speed (25.2 rad/s).
- **5000 has more spike events** (0.29 against 0.19 per episode) and two strict misses.
- **Training trends point away from 5000:**
  - The policy std rises monotonically: 1.75 → 2.16.
  - The share of joint-steps where the limiter clips doubles from 4500 to 5000: 8.5 → 13.4 %.
    The policy increasingly requests jumps that the limiter has to cut.
  - The same std drift came before the 54 % dip at iteration 7000 in the source run.
- **4500 is therefore preferred**, and **5000 is the alternative** if joint-limit penetration
  is the priority.
- "Selected" here means the best by these criteria, not the final checkpoint.

## What did not improve

- **Phase p99 stays at about 15 rad/s:** 15.6 in A for 4500, against 15.1 for the baseline.
  - This is the commanded target slew.
  - The fine-tune makes better use of the allowed slew, so the time spent above 10 rad/s in
    A rises: 12.6 % → 17.9 %.
- **Events in the first 100 ms rose** from 74 to 146 over 1536 episodes, which is still
  < 0.1 per episode. The faster lift is concentrated there.
- **Joint-limit pressing remains.** The targets still sit on a range bound much of the time,
  and 63 % of episodes exceed 0.05 rad somewhere. It is reduced, not solved.
- **The limiter clips more often over training** (5.5 % → 13.4 %).

## Files

| Item | Location |
|---|---|
| Tables | `checkpoint_comparison.csv` (with `*_delta_vs_ref` against the baseline), `per_pose_results.csv`, `per_seed_results.csv`, `joint_velocity_statistics.csv` (per phase and joint, plus event durations) |
| Verification | `analysis.json`: pairing and per-run limiter checks |
| Metadata | `metadata.json`: checkpoints and sha256, seeds, configuration, provenance |
| Plot | `plots/report.png`: one summary figure (recovery, worst pose, spike events, maximum speed, maximum penetration, median time per checkpoint; 4000 = the zero-shot baseline). |
| Video | None rendered, at the user's request (keep artifacts minimal). To view: `uv run play Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3 --checkpoint-file logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000/model_4500.pt --num-envs 16 --viewer viser`. |
| Checkpoints (gitignored, local) | Selected: `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000/model_4500.pt`. sha256 in `metadata.json`. |

## Next step (recommendation, not started)

- **Handle joint-limit pressing first,** then decide whether to tighten to 0.25. The reasons:
  - At 0.3 the fine-tune already reached its goals: 100 % recovery, −78 % spike events,
    −37 % peak penetration.
  - The limiter now clips 8–13 % of joint-steps, and the policy std keeps rising. More
    adaptation at 0.3 has little left to gain.
  - Penetration is the remaining weakness: up to 0.24 rad, and in 63 % of episodes above
    0.05 rad.
  - Tightening to 0.25 would lengthen the loaded B/C phases, the condition under which
    penetration grew in the zero-shot evaluation.
- **A sensible order:**
  1. A target margin inside the joint range, or a penalty, as a separate, single change.
  2. Evaluate with this protocol.
  3. Only then try a 0.25 fine-tune, starting from the selected checkpoint.
- **Not hardware-ready.** The motor speed and inertia limits of the real actuators are
  unverified. The URDF lists 21 rad/s, and the simulated peaks reach 25–30 rad/s.
