# Zero-shot evaluation of a 0.3 rad/step PD-target rate limiter

Evaluation only: no training, no fine-tuning, no change to weights, optimizer states,
rewards, the motor model or the checkpoints. Code: `src/minipi_getup/ftsr_ref/limiter_eval.py`
and `tools/ftsr_limiter_eval.sh`, commit 8878bbc on `ftsr-only`, which is local and not pushed.

## Protocol

- **Checkpoints:** `model_3000`, `model_4000`, `model_5000` and `model_8000` of
  `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700`.
  The sha256 of each is in `metadata.json`.
- **Task:** `Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless`, the v2 stateless stages.
  - Play cfg: no observation noise.
  - **Zero assistance.** Deterministic student (mean action).
  - Physics: 0.5 ms × 40.
  - Episode: 20 s, which is a 2 s passive window followed by 18 s of policy control.
  - Plant: PD with kp/kd as in training, ±16 Nm.
- **Episodes:** 4 poses (supine, prone, left_side, right_side), 128 episodes each.
  - Three seeds: 2150, 2151 and 2152. That is 1536 episodes per checkpoint and configuration.
  - This is the same rollout as the periodic eval (`analyze_recovery periodic`), with more episodes.
- **Pairing:** the baseline and limited runs of a seed start from identical states. The maximum
  |Δ qpos₀| over all runs of a seed is **0.0** (`analysis.json`). The initial states are also
  identical across checkpoints.
- **Recovery** uses `analyze_recovery.analyze`, unchanged.
  - Recovered: over the last 3 s, mean base height > 0.19 m and mean uprightness > cos 18°.
  - Strict: base height > 0.31 m and upright during the whole of the last 3 s.
  - Recovery time: the start of the first 1 s held above h1 and upright, measured after the
    passive window.
- **Phases** use the velocity-audit definitions:
  - A: from the end of the passive window to the first h1 crossing.
  - B: h1 → first h2 crossing.
  - C: h2 → the start of the first 0.5 s above h2 and upright.
  - D: after that.
- **Joint speed:** recorded at every 0.5 ms physics step, quantized to 0.01 rad/s. Per-phase and
  per-joint histograms are exact to 0.01 rad/s.
  - Event: a run of consecutive physics steps of one joint with |qdot| > 20 rad/s.
- **Physical safety:**
  - Joint-limit penetration: the amount by which q is outside the XML range.
  - |tau| and saturation (≥ 0.95 × 16 Nm).
  - Maximum |q* − q|.
  - Root constraint force: the contact proxy, in body weights.
  - Invalid / NaN states.

### Limiter (configuration B)

```
q_clip_t = clip(q_default + scale * clip(a_t, ±50), q_min, q_max)   # unchanged pipeline
q*_t     = q*_{t-1} + clip(q_clip_t - q*_{t-1}, -0.3, +0.3)          # per joint, 50 Hz
q*_{t-1} := clip(q_measured_t, q_min, q_max)  in the passive window and at the first
            actuated step of an episode
PD:  tau = clip(kp (q*_t - q) - kd qdot, ±16)                         # unchanged
```

- **Implementation:** it wraps `FtsrAction.process_actions` in the evaluation process. It runs
  after the original pipeline, so the raw clip, the physical clip and passive handling are
  unchanged. It replaces only `q*`, which `apply_actions` holds for the 40 physics steps.
- **Untouched:** the raw action, `last_action` in the observation, the network, the gains, the
  torque cap and the joint ranges.
- **Verified at every policy step** of all 12 limited runs (`analysis.json → runs.*.limiter_check`):
  - Maximum |q*_t − q*_{t−1}| = 0.3000001 rad. The excess is float32 rounding.
  - The limiter is active on 4–6 % of actuated joint-steps (13 % for model_8000).
  - 0 targets fall outside the joint range.
  - The first actuated target is ≤ 0.300 rad from the measured pose. All 512 entries in each
    run were checked.
  - Raw-action mismatch is 0.0.

## Main comparison (3 seeds pooled, 1536 episodes per row)

| Checkpoint | Limiter | Recovery | Worst pose (min over seeds) | Recovery time median / p95 (s) | Lifting qd p99 (rad/s) | qd max (rad/s) | Events > 20 per episode | Joint violation max / episodes > 0.05 rad |
|---|---|---|---|---|---|---|---|---|
| model_3000 | none | 99.6 % | 99.2 % (98.4) | 0.60 / 3.20 | 26.0 | 49.5 | 17.4 | 0.301 / 94 % |
| model_3000 | 0.3 | 93.4 % | 90.4 % (88.3) | 3.22 / 11.04 | 13.4 | 43.4 | 0.90 | 0.585 / 92 % |
| model_4000 | none | 100 % | 100 % (100) | 0.34 / 1.55 | 30.2 | 48.6 | 14.6 | 0.287 / 83 % |
| **model_4000** | **0.3** | **98.2 %** | **97.7 % (96.9)** | **1.34 / 7.67** | **15.2** | **34.3** | **0.91** | 0.440 / 87 % |
| model_5000 | none | 100 % | 100 % (100) | 0.28 / 1.42 | 29.9 | 46.7 | 12.8 | 0.246 / 76 % |
| **model_5000** | **0.3** | **98.4 %** | **97.4 % (96.1)** | **2.00 / 7.20** | **15.0** | **32.0** | **0.73** | **0.293 / 80 %** |
| model_8000 | none | 99.9 % | 99.5 % (98.4) | 0.36 / 1.53 | 29.7 | 49.9 | 14.9 | 0.169 / 79 % |
| model_8000 | 0.3 | **50.7 %** | 24.0 % (21.9) | 4.86 / 13.62 | 15.2 | 35.0 | 1.51 | 0.323 / 96 % |

Per seed (`per_seed_results.csv`), limited recovery:

| Checkpoint | Seed 2150 | Seed 2151 | Seed 2152 |
|---|---|---|---|
| model_3000 | 93.8 | 92.8 | 93.8 |
| model_4000 | 98.2 | 98.2 | 98.0 |
| model_5000 | 98.2 | 98.6 | 98.4 |
| model_8000 | 48.2 | 52.3 | 51.6 |

The seed-to-seed spread is ≤ 4 points.

### Changes relative to each checkpoint's own baseline

| Checkpoint | Events > 20 rad/s | First-100 ms events | Lifting p99 | qd max | Joint violation max | Median recovery time |
|---|---|---|---|---|---|---|
| model_3000 | −94.8 % | 11434 → 132 | −48.6 % | −12.3 % | +94 % | +2.62 s |
| model_4000 | −93.8 % | 8201 → 71 | −49.7 % | −29.4 % | +53 % | +1.00 s |
| model_5000 | −94.3 % | 7541 → 37 | −49.8 % | −31.5 % | +19 % | +1.72 s |
| model_8000 | −89.8 % | 7941 → 44 | −48.9 % | −29.9 % | +91 % | +4.50 s |

### Recovery funnel (3 seeds pooled, %)

| Checkpoint | Limiter | Reached h1 | h2 given h1 | Recovered given h2 | Recovered | Strict > 0.31 m | Foot-only stable |
|---|---|---|---|---|---|---|---|
| model_3000 | none / 0.3 | 99.8 / 98.4 | 99.9 / 98.3 | 99.9 / 96.6 | 99.6 / 93.4 | 99.6 / 92.6 | 99.6 / 93.3 |
| model_4000 | none / 0.3 | 100 / 99.9 | 100 / 99.8 | 100 / 98.5 | 100 / 98.2 | 100 / 97.4 | 100 / 98.2 |
| model_5000 | none / 0.3 | 100 / 100 | 100 / 99.9 | 100 / 98.5 | 100 / 98.4 | 99.5 / 97.6 | 100 / 98.4 |
| model_8000 | none / 0.3 | 100 / 99.7 | 100 / 74.4 | 99.9 / 68.4 | 99.9 / 50.7 | 91.0 / 44.4 | 99.9 / 50.5 |

Failures with the limiter (3 seeds):

| Checkpoint | Failure breakdown |
|---|---|
| model_4000 | high-but-tilted 1.1 %, reached h1 but not held 0.5 % |
| model_5000 | high-but-tilted 0.8 %, reached h1 but not held 0.4 %, fell after recovery 0.4 % |
| model_3000 | reached h1 but not held 3.6 %, high-but-tilted 1.4 %, never lifted 0.8 % |
| model_8000 | **reached h1 but not held 45.4 %**: it stalls between h1 and h2. Phase D also has p99 8.9 rad/s, against 4.9 without the limiter. |

## Joint velocity by phase (3 seeds pooled, all joints, rad/s)

| Checkpoint | Limiter | A p99 / max | B p99 / max | C p99 / max | D p99 / max | First 100 ms p99 / max | Fraction > 20 in A |
|---|---|---|---|---|---|---|---|
| model_3000 | none | 26.0 / 49.5 | 24.2 / 48.2 | 20.5 / 45.4 | 4.4 / 43.5 | 31.2 / 49.5 | 3.0 % |
| model_3000 | 0.3 | 13.4 / 31.8 | 14.4 / 33.6 | 14.4 / 43.4 | 4.6 / 31.9 | 15.6 / 26.6 | 0.009 % |
| model_4000 | none | 30.2 / 48.6 | 25.1 / 43.5 | 20.5 / 45.8 | 6.2 / 38.7 | 30.7 / 48.6 | 7.5 % |
| model_4000 | 0.3 | 15.2 / 28.7 | 15.4 / 34.3 | 15.1 / 33.4 | 5.8 / 29.2 | 15.2 / 24.9 | 0.018 % |
| model_5000 | none | 29.9 / 46.7 | 20.8 / 39.9 | 20.1 / 45.8 | 4.3 / 43.6 | 30.4 / 46.7 | 7.8 % |
| model_5000 | 0.3 | 15.0 / 28.5 | 15.3 / 32.1 | 14.7 / 31.4 | 4.5 / 27.1 | 15.0 / 25.3 | 0.006 % |
| model_8000 | none | 29.7 / 49.9 | 18.9 / 38.0 | 19.4 / 44.3 | 4.9 / 38.0 | 30.5 / 46.0 | 8.9 % |
| model_8000 | 0.3 | 15.2 / 27.3 | 15.4 / 34.9 | 14.8 / 35.0 | 8.9 / 27.4 | 15.2 / 27.3 | 0.007 % |

- **The limiter caps the phase p99 at about 15 rad/s.** This is the commanded target speed:
  0.3 rad per 20 ms. The cap holds in every phase.
- **Peaks of 25–35 rad/s remain.** They come from contact and inertial coupling, and from
  tracking an error that has accumulated. Per-joint and per-phase values, including event
  durations, are in `joint_velocity_statistics.csv`.
- **Whole-episode fractions above 6.28 and 10 rad/s rise with the limiter** (`checkpoint_comparison.csv`).
  - The cause: phases A–C last 2–5× longer, at about 10–15 rad/s, instead of a short burst at
    20–50 rad/s.
  - Inside each phase, the fraction above 10 rad/s drops by about half in A, and the fraction
    above 20 rad/s by 99.7–99.9 %.
  - This is why the whole-episode p99 must not be used to judge the limiter.

## Physical safety

- **No invalid state, NaN or physics explosion** in any of the 14 336 episodes: 24 main runs
  plus 4 threshold-scan runs, 512 episodes each. There is no simulator instability to report separately.
  Every high-speed sample here is valid motion.
- **Torque:** 16.0 Nm (the cap) is reached in every configuration.
  - Saturation is 0.24–0.51 % of joint-samples without the limiter and 0.44–0.94 % with it.
  - It is higher with the limiter because the get-up lasts longer.
- **Maximum tracking error |q* − q|** drops from 2.5–2.7 rad to 2.0–2.2 rad.
- **Root constraint force (contact proxy):**
  - Peaks of 18–22 body weights, with or without the limiter.
  - Episodes with a peak above 10 body weights: 61–65 % with the limiter for 3000–5000, and
    68–88 % without.
- **Joint-limit penetration gets worse at the maximum.** The episode fraction above 0.05 rad
  is about the same.
  - Limiter maxima: 0.29–0.59 rad. Without: 0.17–0.30 rad.
  - The largest values are on the low-gain joints: thigh, i.e. hip yaw (kp 20), and ankle roll
    (kp 10). They occur in phases B and C, under body load.
  - The share of phase-A policy steps above 0.05 rad falls, e.g. for model_4000 from 16.9 % to
    9.1 %. The share in B/C is similar or lower.
  - In 58–67 % of actuated joint-steps, the PD target sits on a range bound, with or without the
    limiter.
  - This is the static pressing documented in the velocity audit. The limiter does not cause it
    and does not fix it. It lengthens the time spent loaded in B/C, which raises the maximum.

## Threshold scan (zero-shot, seed 2150, 512 episodes each, `threshold_scan/`)

| Checkpoint | Limit (rad/step) | Recovery | Worst pose | Median / p95 time (s) | Events > 20 per episode | qd max | Joint violation max |
|---|---|---|---|---|---|---|---|
| model_4000 | 0.30 | 98.2 % | 96.9 % | 1.34 / 7.67 | 0.89 | 34.3 | 0.440 |
| model_4000 | 0.25 | 88.1 % | 84.4 % | 1.84 / 11.67 | 0.50 | 32.4 | 0.417 |
| model_4000 | 0.20 | 70.9 % | 53.9 % | 2.46 / 13.55 | 0.17 | 27.7 | 0.462 |
| model_5000 | 0.30 | 98.4 % | 96.1 % | 2.00 / 7.20 | 0.72 | 32.0 | 0.293 |
| model_5000 | 0.25 | 60.4 % | 28.9 % | 2.62 / 11.78 | 1.31 | 34.3 | 0.214 |
| model_5000 | 0.20 | 48.2 % | 5.5 % | 3.98 / 12.17 | 0.25 | 30.3 | 0.172 |

The 0.3 rows give recovery, time, qd max and violation pooled over 3 seeds from the main table,
and events per episode for seed 2150.

- **0.3 is the tightest limit that keeps ≥ 95 % zero-shot.**
- **model_4000 degrades far more gracefully than model_5000** when the limit tightens.

## Files

| File | Contents |
|---|---|
| `checkpoint_comparison.csv` | One row per checkpoint × limiter, pooled over seeds, with every metric above and the change vs. baseline. |
| `per_pose_results.csv` | Per checkpoint × limiter × seed × pose: recovery, strict, foot-only, funnel, times, events, safety, failure categories. |
| `per_seed_results.csv` | Seed-level recovery, worst pose and events. |
| `joint_velocity_statistics.csv` | Per phase (A–D and the first 100 ms) × joint: samples, p95 / p99 / max and fractions above 4 / 6.28 / 10 / 20 rad/s. Also > 20 rad/s events per joint with mean and maximum duration. |
| `analysis.json` | Pairing check and per-run limiter verification. |
| `metadata.json` | Checkpoints and hashes, seeds, git SHA and configuration. |
| `plots/` | Bars for recovery, worst pose, events, lifting p99, qd max, time and violations; the lifting-phase CCDF; paired traces. |
| `videos/` | Real simulation renders. |
| `runs/`, `threshold_scan/` | Per-run `meta.json` and `stats.json`. `run.npz` is gitignored and local only. |

- **Paired trace plots** (`plots/paired_trace_it*_env*.png`): no limiter on the left, the limiter
  on the right, on the same axes. Rows:
  1. qdot of the joint with the highest baseline peak, with ±20 rad/s marked.
  2. q* (step) and q, with the joint range.
  3. Applied torque, with ±16 Nm marked.
  4. Base height, with h1 and h2.
  5. Maximum joint-limit penetration per step.
- **Videos** (`videos/`, mp4 files are gitignored):
  - `it5000_success_limited_*`
  - `it5000_failed_limited_*`
  - `it5000_paired_*`: side-by-side, the same env and initial state with and without the limiter.
  - The same set for model_4000.
  - model_8000 failure examples.
  - Real time and 0.25× slow motion.
