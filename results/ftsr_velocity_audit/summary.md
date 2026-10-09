# Mini-Pi FTSR recovery: joint-velocity audit (before fine-tuning)

Date: 2026-10-09. Branch `ftsr-only`. The audit is simulation-only and evaluation-only. No training code, reward, plant, gain or action pipeline was changed. Training (`ftsr_recovery_pd16_noslew_resume2700`) was never stopped. `metadata.json` holds the exact checkpoints (with sha256), seed, task, simulation configuration and commands.

## Setup

**Evaluation**

| | |
|---|---|
| Task | `Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless`, play cfg |
| Assistance | **zero** |
| Policy | deterministic student |
| Seed | 2150 |
| Episodes | 128 per checkpoint: 32 per fallen pose (supine, prone, left, right), the same resets for every checkpoint |
| Episode length | 20 s: 2 s passive window, then 18 s of policy |

**Recording**

- *Every 0.5 ms physics step:* q, qdot, applied torque, the PD torque before clipping, the joint-space constraint force (contact + joint limit), base height, uprightness and ground-contact groups.
- *Every 20 ms policy step:* q*, the policy command before range clipping, every reward term and validity flags.
- *Checks on the recording:* the applied torque equals clip(PD, ±16) to within float16 rounding (≤ 0.008 Nm). The logged `qd_soft_envelope` equals the formula applied to the last physics step of each policy step (error ≤ 6e-6).

**Checkpoints**

| Checkpoint | Note |
|---|---|
| model_2500 | |
| model_3000 | best autonomous checkpoint (periodic eval 99.6 %) |
| model_3500, 4000, 4500 | |

All five use the same pd16_noslew plant: PD clipped at ±16 Nm, no torque-speed envelope, no target slew. Results from earlier plants (v0–v2) are not compared here.

**Counterfactuals** (*evaluation only, not the training plant*): model_3000 re-run with a PD-target rate limit of 0.3, 0.2 and 0.1 rad per policy step.

## Model facts that matter (verified in the runtime model)

- **Joints:** armature 0, joint damping 0, frictionloss 0.
- **Joint limits:** MuJoCo's default soft constraint (solref 0.02 s, solimp 0.9/0.95).
- **PD gains:** kp 60/40/20/60/30/10, kd 2.4/0.8/0.4/2.8/1.6/0.3 (hip pitch, hip roll, thigh, calf, ankle pitch, ankle roll).
- **Large tracking error turns the PD into a velocity servo.** The PD wants qdot ≈ (kp/kd)·(q* − q). For hip pitch kp/kd = 25 s⁻¹, so a 2 rad error asks for about 50 rad/s.
- **Nothing limits that speed.** There is no rotor inertia, no torque-speed derating and no slew. The torque cap only limits the acceleration.
- **Free-limb joint inertia is tiny.** From the mass matrix at the default pose with all other joints free, 1/(M⁻¹)ᵢᵢ is:

| Joint | Inertia (kg m²) | Time to 20 rad/s at 16 Nm |
|---|---|---|
| hip pitch | 0.0041 | ≈ 5 ms |
| calf | 0.0029 | ≈ 3.6 ms |
| ankle roll | 0.0004 | ≈ 0.5 ms |

## Results per checkpoint (zero assistance, 128 episodes each)

| it | recovered (supine / prone / left / right) | strict > 0.31 m | median time from policy start to h1 / h2 / stable (s) | max \|qd\| (rad/s) | events > 20 rad/s | invalid episodes |
|---|---|---|---|---|---|---|
| 2500 | 28 / 22 / 44 / 25 % | same | 1.64 / 5.46 / 5.18 | 41.8 | 4768 | 0 |
| **3000** | **100 / 100 / 100 / 100 %** | 100 % | 0.22 / 0.40 / 0.58 | 46.9 | 2185 | 0 |
| 3500 | 94 / 97 / 100 / 100 % | 94–100 % | 0.28 / 0.40 / 0.44 | 45.3 | 2165 | 0 |
| 4000 | 100 % all | 100 % | 0.21 / 0.30 / 0.37 | 46.1 | 2050 | 0 |
| 4500 | 100 % all | 100 % | 0.22 / 0.30 / 0.34 | 47.2 | 1845 | 0 |

The 3 failed episodes at 3500 stood up, then fell later.

**Trend from 3000 to 4500:** the get-up becomes faster and its joint speeds rise.
- Phase A p95 \|qd\| goes from 18.2 to 22.7 rad/s.
- The share of phase-A physics steps above 10 rad/s goes from 15 % to 28 %.
- The whole get-up shrinks from 0.58 s to 0.34 s (median).

## Answers

### 1. Are the 38–46 rad/s peaks valid policy motion or physics explosions?

**Valid policy motion.**
- In 1024 audited episodes there were no non-finite states, no `nan` terminations and no invalid flags.
- The 46.9 rad/s maximum (model_3000, env 13, right hip pitch, t = 2.010 s) happens in a successful episode, in the **first policy step after the passive window**. The policy's first target jumps from −0.43 rad to the joint's upper range limit of 1.75 rad, a 2.2 rad step (`plots/spike_events_it3000.png`, row 1).
- The single physics explosion seen so far happened in training at iteration 2700. It was an isolated event, already handled by the non-finite guard (commit 9a6796a), and has not recurred.

### 2. Which joints and phases are responsible?

**Joints.**
- Right and left hip pitch and calf, then ankle pitch and right ankle roll (`joint_velocity_statistics.csv`).
- At model_3000, right hip pitch has p99 11.6 rad/s, max 46.9, and 3.0 episodes per episode above 20 rad/s (median 30 ms long). Right calf has p99 14.2, max 39.2, and 4.1 events per episode.
- Hip roll never exceeds 20 rad/s.

**Phases** (`phase_statistics.csv`, `plots/qd_by_joint_phase_it3000.png`).

| Phase at model_3000 | median duration | p95 / p99 \|qd\| | share of all \|qd\| > 10 rad/s samples |
|---|---|---|---|
| A, lift (to h1) | 0.22 s | 18.2 / 27.4 | **48 %** (54 % at 4000 and 4500) |
| B, h1 → h2 | 0.15 s | 17.3 / 25.4 | 21 % |
| C, stabilize | 0.20 s | 11.7 / 20.2 | 21 % |
| D, standing | 17.4 s | 2.0 / 4.3 | 10 % |

- **The first 100 ms after the passive window** alone contain 42 % of all > 20 rad/s events at 3000 (34 % at 4500).
- **During standing (D)** speeds are moderate: p99 4.3–7.2 rad/s across checkpoints.

### 3. Are large q* jumps the main cause?

**Yes, for almost every spike.** The evidence is a dose-response at the physics-step level (`target_statistics.csv`, model_3000, phases A–C, each joint and policy step):

| \|Δq*\| in one 20 ms step | share of joint-steps | median peak \|qd\| in the next 40 ms | P(peak > 20 rad/s) |
|---|---|---|---|
| < 0.05 rad | 73 % | 2.2 | 1.9 % |
| 0.05–0.15 | 7.5 % | 5.4 | 4.7 % |
| 0.15–0.25 | 5.3 % | 7.6 | 6.3 % |
| 0.25–0.5 | 7.6 % | 11.1 | 13 % |
| 0.5–1.0 | 5.2 % | 16.2 | 34 % |
| ≥ 1.0 | 1.7 % | 22.8 | **61 %** |

**Event classification** (2185 events at 3000):

| Cause | Events |
|---|---|
| ≥ 0.25 rad target step of that joint toward the motion within 60 ms before the peak | 2170 |
| actuator-driven without such a jump | 10 |
| joint-limit bounce | 5 |
| contact impulse | 0 |
| explosion | 0 |

- The base rate of such a step in any 60 ms window is 29 %, while 99 % of spike events have one. The association is therefore not a base-rate artefact.
- At 4500, 1841 of 1845 events follow such a step.

**Mechanism, as seen in the traces.**
1. The target jumps (median 1.03 rad over the 60 ms before the peak). In 82–84 % of events the jump alone would saturate the PD.
2. The torque saturates for a few ms and the near-zero-inertia limb accelerates within about 10–20 ms (median 22.5 ms from 6.28 rad/s to the peak).
3. The speed peaks when the PD damping term balances the remaining error. Only 4 % of peaks occur while the torque is at the cap.

**Target behaviour.**
- The policy is close to bang-bang: during lift, targets sit at the joint-range clip 26–97 % of the time per joint (hip roll 96 %). The policy command is outside the range just as often.
- In the first policy step, both hip pitch targets jump by a median 1.5 rad (98 % of episodes ≥ 0.25 rad).
- Contact onsets within ±100 ms occur around 83–93 % of events. Get-up contact changes constantly, so this does not discriminate a cause. The constraint impulse dominates in under 1 % of events.

### 4. Is the current velocity penalty effective during recovery?

**No.**

**Sampling.** `qd_soft_envelope` = −0.01·Σ relu(\|qd\| − 6.28)² reads `joint_vel` once per policy step, i.e. the state after the last of 40 physics steps (50 Hz).
- It misses 26 % (it 3000) to 43 % (it 4000/4500) of physics steps above 6.28 rad/s.
- Evaluated at every physics step it would be 1.3–1.5× larger.

**Stages.** It applies in all stages (−0.01 in r_u, r_s and r_w). Invalid physics samples are excluded in training by the runner guard (commit 9a6796a); none occurred here.

**Magnitude during recovery** (model_3000, rewards with the training stage r_w, per second):

| Phase | total reward | qd_soft_envelope | dof_vel | dof_acc | action_rate | orientation | height |
|---|---|---|---|---|---|---|---|
| A, lift | −24.5 | −2.08 | −2.34 | −2.74 | −4.45 | −9.52 | +0.09 |
| B | −14.7 | −1.78 | −2.13 | −2.10 | −3.62 | −2.89 | +0.61 |
| C | −10.2 | −0.63 | −1.00 | −0.94 | −2.09 | −5.11 | +1.92 |
| D, standing | **+10.8** | −0.003 | −0.04 | −0.05 | −0.06 | +1.46 | +4.72 |

**Trade-off.**
- Every second spent before standing costs about 35 reward relative to standing (−24.5 vs +10.8).
- The entire qd envelope penalty of one get-up is about 1.6 reward (phase sums; all velocity-type terms together about 5.7). That is the value of standing up 0.05 s earlier.
- The quadratic velocity penalties scale like Δ²/τ for a motion of amplitude Δ done in time τ, while the time cost scales like 35·τ. The optimum τ ∝ √(weight), so only large weight increases (order 10–100×) would move it noticeably.
- The training trend (faster and faster get-ups from 3000 to 4500 at an unchanged weight) is consistent with this.

### 5. Smallest justified change that reduces joint speed while preserving recovery

**A PD-target rate limiter of 0.3 rad per policy step (15 rad/s of target motion).** Rewards stay unchanged. It acts on the measured cause, the target step, and it bounds the velocity request itself instead of penalizing it after the fact.

Zero-shot on model_3000, without any retraining:

| | recovered | > 20 rad/s events | p99 \|qd\| A / B / C | max | median time to stable |
|---|---|---|---|---|---|
| baseline | 100 % | 2185 | 27.4 / 25.4 / 20.2 | 46.9 | 0.58 s |
| limit 0.3 rad/step | **94 %** (100 / 88 / 91 / 97) | **97 (−96 %)** | **13.5 / 14.3 / 14.4** | 35.3 | 2.8 s |
| limit 0.2 rad/step | 8 % | 5 | 9.7 / 10.6 / 10.2 | 20.8 | 6.8 s |
| limit 0.1 rad/step | 0 % | 0 | 5.5 / – / – | 19.4 | – |

- The fraction of samples above 6.28 rad/s rises under the 0.3 limit, because the get-up lasts longer.
- The extreme part disappears: events above 20 rad/s fall by 96 % and every phase's p99 drops to about 14 rad/s.
- The limiter changes the action contract: deploy must apply the same limiter. Details are in `recommendation.md`.

### 6. Fine-tune from the best checkpoint, or retrain?

**Fine-tune from model_3000.** It keeps 94 % recovery under the 0.3 limit with no training at all, so the policy only needs to adapt, not relearn.
- A from-scratch retrain is not justified by the evidence.
- A limit of 0.2 rad/step drops recovery to 8 % zero-shot. It should only be reached by tightening gradually during fine-tuning, after 0.3 has been re-converged.

## Joint-limit violations (`joint_limit_statistics.csv`)

These are physical joint positions beyond the XML range by more than 5 mrad, not the clipping of commanded targets.

**Where (model_3000).**
- The most frequent joints are right thigh (20 % of valid physics steps), left and right hip roll (7–13 %), right calf (13 %) and left thigh (9 %).
- The largest are right ankle roll (max 0.19 rad, p99 of violating samples 0.12) and left thigh (max 0.17).
- Similar at 4500: right calf 25 %, left hip roll 20 %, max 0.17 rad.

**Cause: a static press, not an impact.**
- The policy actively drives into the limit. While the joint is past its limit, the policy's target sits exactly at that limit and its command lies beyond it in 99–100 % of samples for the joints with the most violations (thigh, hip roll, right calf, ankle roll) and in 83–96 % for the others.
- At model_3000, at most 0.2 % of violation samples per joint happen with \|qd\| > 10 rad/s.
- Most happen in phase D, standing (e.g. right thigh: 861k of 940k violation samples).
- The default soft limit (solref 0.02 s) lets a 16 Nm press sink 0.1–0.19 rad.
- The `pen_dof_pos_limits` penalty (−1.2 to −1.7 per s in every phase) does not stop it.

This is a separate issue from the velocity spikes, and a rate limiter will not fix it. See `recommendation.md`.

## Files

| File | Contents |
|---|---|
| `metadata.json` | checkpoints (sha256), seed, task, simulation config, definitions, commands |
| `joint_velocity_statistics.csv` | per run, episode class (success / failed / invalid / all valid) and joint: mean, p95, p99 and max \|qd\|; fractions above 4 / 6.28 / 10 / 20 rad/s; event counts and durations above 6.28 and 20 rad/s |
| `phase_statistics.csv` | per run, class and phase: durations, completion rate, qd percentiles, torque, limit violations, top joints, and every reward term per second (stage r_w) with phase sums |
| `spike_events.csv` | every event above 20 rad/s: joint, time, phase, q, qd, q*, command before clipping, tracking error, torque, PD torque before clipping, cap flag, actuator and constraint impulses during the rise, target jumps (own and same-leg), limit penetration, contacts, base state, validity, cause |
| `target_statistics.csv` | per run, phase and joint: target-step distribution, time at the range clip, commands beyond the range (first policy step separate) |
| `joint_limit_statistics.csv` | per run and joint: violation fraction, max, p99, side, relation to speed and to target clipping, phase counts |
| `analysis.json` | per-run summaries, `qd_soft_envelope` sampling check, diagonal inertia estimates from data (noisy, superseded by the model mass-matrix values above) |
| `plots/` | joint speed, q / q*, applied and PD torque with constraint force, and base height vs time, for the high-speed and low-speed successes at 3000 and 3500 and the failure at 3500 (full and zoomed, with h1 / h2 / stable, contact onsets, limit violations and the peak marked); ±100 ms traces of representative spike events per cause; joint × phase heatmaps; iteration comparison; spike causes by iteration |
| `videos/` | rendered from the recorded qpos of those episodes, at real time and at 0.2× for the get-up (git-ignored mp4s, kept locally) |

The low-speed success still peaks at 27.8 rad/s (3000) or 25.8 rad/s (3500). Every successful episode has at least one spike above 20 rad/s. No explosion video exists, because no explosion occurred in 1024 evaluation episodes.

**Raw data:** `runs/*/rollout.npz`, about 0.9 GB each, git-ignored.
