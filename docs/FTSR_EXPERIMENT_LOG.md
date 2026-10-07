# FTSR experiment log (Mini-Pi, MjLab)

Append-only. Failed and rejected runs stay in the log.

Common setup unless noted:

- RTX 3090, mjlab 1.6.0, 2 ms physics × decimation 10 (50 Hz policy), flat ground.
- Runs live in `logs/rsl_rl/minipi_ftsr/<timestamp>_<run name>/`. Each run dir has
  `git_state.txt` (HEAD, status, diff at launch) and `params/`.
- Debug and smoke runs (Phase A) are not experiments and are not listed.

---

## ftsr_pretrain_rw_v1 (Phase C, r_w pretraining)

- Date: 2026-10-07 01:56 (+07)
- Code: `fe42ec2`, clean tree
- Task: `Mjlab-FTSR-MiniPi-Walk`. Standing resets, walking stage fixed, no assist,
  fall termination. Same networks and observations as the full task.
- Init: random. Envs 4000 (3000 teacher / 1000 student), seed 42, 200 iterations.
- Parameters: stage r_w weights (Table II as adapted), action scale 0.13 relative,
  9 Nm operational envelope, full DR.
- Purpose: paper Sec. III-A, "Model Initialization": elementary walking in ~200
  iterations.
- Expected: tracking emerges and falls drop; torque stays well under 9 Nm.
- Result (walk eval, 2 seeds × 200 envs, 10 s, student policy, `logs/ftsr_analysis/pretrain_walk.csv`):

  | it | falls | lin err (moving) | yaw err | joint vel | torque p99 / max | sat |
  |---|---|---|---|---|---|---|
  | 100 | 0.25 % | 0.175 | 0.144 | 0.016 rad/s | 1.9 / 6.5 Nm | 0 |
  | 150 | 0 | 0.175 | 0.140 | 0.015 | 2.2 / 7.6 | 0 |
  | 200 | 0 | 0.175 | 0.140 | 0.013 | 2.8 / 8.1 | 0 |

- **Outcome: rejected.** The policy stands still and balances well, but doesn't walk.
  The moving-command error equals the mean commanded speed, feet air time dropped to
  0.006 s, and joint speed is 0.013 rad/s. Cause: under the Froude-scaled tracking
  kernel exp(-18 dv^2), standing still keeps about half the tracking reward. The
  training reward rose (213), but that is not elementary walking.

## ftsr_pretrain_rw_v2 (Phase C, retry)

- Date: 2026-10-07 02:09. Code `649d887`. Init random, 4000 envs, seed 42, up to 400 it.
- Change (one family, locomotion shaping): lin-vel kernel exp(-25 dv^2) (std 0.2 m/s,
  the width of the tuned Mini-Pi velocity task), feet air time weight 1.0 → 2.0.
- Expected: stepping emerges and the moving-command error drops clearly below the
  command magnitude.
- Result (walk eval, same protocol): it 200 / 300 / 400 all give a moving lin-vel
  error of 0.175 and a yaw error of 0.140, equal to the mean command magnitudes. A
  debug rollout with random commands (200 envs) found zero command/velocity
  correlation (vx r = -0.02, wz r = 0.09). Feet air time ~0 throughout training;
  tracking reward flat after it 200.
- **Outcome: rejected.** Still a standing-still optimum. Exploration in relative joint
  deltas never produces coherent alternating steps; JiaRan is wheeled and never had
  to discover a gait.
- Side finding: mjlab `ObservationManager.compute()` returns a cached buffer, so
  the evaluator's first observation after a command override was stale. Fixed in
  `3f14ee0`. It didn't affect any metric, which is averaged from 1 s on.

## ftsr_pretrain_rw_v3 (Phase C, retry 2)

- Date: 2026-10-07 02:25. Code `3f14ee0`. Init random, 4000 envs, seed 42, up to 400 it.
- Change (gait shaping, the legged replacement of the wheeled locomotion terms):
  `[sin, cos]` gait clock (period 0.5 s, zero for stand commands) added to o_t
  (o_t 45 → 47, student 225 → 235, critic 49 → 51), and `feet_gait` reward 1.5 in
  r_w only (anti-phase legs, 55 % stance). Values from the tuned Mini-Pi velocity
  task.
- Expected: alternating steps; the moving-command error clearly below 0.175.
- Result (walk eval): it 200 / 300 / 400 → moving lin-vel error 0.180 / 0.182 /
  0.181, yaw 0.15. Joint speed rose to 0.06–0.10 rad/s, but air time stayed at
  0.01–0.03 s and `feet_gait` at the both-feet-down chance level (0.47). Torque p99
  ≤ 4 Nm, no saturation.
- **Outcome: rejected.** Still no stepping, so the gait clock alone isn't enough.

## diag_walk_absolute (diagnostic, not a candidate)

- Date: 2026-10-07 02:40. Code `3f14ee0` plus a `relative` switch in the action term
  (CLI: `--env.actions.joint-pos.relative False --env.actions.joint-pos.scale 0.25`).
  4000 envs, 300 it.
- Hypothesis: the relative-target action (my deviation from the paper's
  `q_default + scale·a`) is what blocks gait discovery. Exploration noise integrates
  into a random walk of the posture, and the policy learns to freeze. The sibling
  Mini-Pi velocity task walks with absolute targets at 0.25 rad.
- Result (walk eval, 2 seeds × 200 envs): it 300 → lin-vel error 0.049 (moving
  0.060), yaw 0.24, command success 60 %, no falls, joint speed 0.64 rad/s, torque
  p99 5.7 Nm, max 9.4, saturation 0.
- **Conclusion: hypothesis confirmed.** With absolute targets the robot walks within
  300 iterations; with relative targets it never did (3 runs). FTSR moves to the
  paper's absolute action form.

## diag_walk_absolute013 (diagnostic)

- Same as above with absolute scale 0.13 (the requested starting scale) and
  `clip_actions` 12, so targets can reach the get-up joint ranges (±1.56 rad around
  the stance). Question: does the smaller, smoother scale still learn to walk?
- Result: it 300 → moving lin-vel error 0.167, yaw 0.20, command success 20 % (the
  zero-command envs only), torque p99 4.0 Nm. **No walking at 0.13 within 300 it.**

### Decision after the Phase C diagnostics

Absolute targets at 0.25 rad, `clip_actions` 7, operational envelope unchanged at
9 Nm (commit `8be792b`). It is the paper's action form and walks within 300 it in
this setup. The user-suggested 0.13 didn't walk (relative or absolute).
Smoothness and torque are handled by the 9 Nm cap, the Table II rate/smoothness
terms (on raw actions, so at 0.25 they're stricter per radian than on JiaRan at
0.6), and the monitored saturation fraction.

## ftsr_pretrain_rw_v4 (Phase C, final pretraining config)

- Date: 2026-10-07 03:00. Code `8be792b`, clean. Init random, 4000 envs (3000/1000),
  seed 42, 300 it. Absolute 0.25, clip 7, gait clock + `feet_gait`, lin-vel kernel
  std 0.2, air time 2.0.
- Result (walk eval, 2 seeds × 200 envs, `logs/ftsr_analysis/pretrain_walk.csv`):

  | it | falls | lin err (moving) | yaw err | cmd success | torque p99 / max | sat | action rate |
  |---|---|---|---|---|---|---|---|
  | 150 | 0 | 0.067 | 0.443 | 21 % | 5.5 / 9.4 | 0 | 0.24 |
  | 200 | 0 | 0.057 | 0.332 | 25 % | 5.4 / 9.5 | 0 | 0.27 |
  | 250 | 0 | 0.054 | 0.257 | 54 % | 5.0 / 9.4 | 0 | 0.31 |
  | 300 | 0 | 0.053 | 0.180 | 89 % | 4.7 / 9.2 | 0 | 0.36 |

- **Outcome: accepted.** Elementary walking with yaw tracking. Selected
  `/home/huy/minipi_getup/logs/rsl_rl/minipi_ftsr/2026-10-07_02-51-08_ftsr_pretrain_rw_v4/model_300.pt` (sha256 `96b76afdbd7256ff3795e5e568918f41d3d3b0a786c811eba552329b64784a5a`): best tracking, no falls, lowest torque p99.
  Not stopped at 200 because yaw tracking was still poor there (0.33 rad/s).

## ftsr_diag_v1 (Phase B diagnostic, not a candidate)

- Task `Mjlab-FTSR-MiniPi`, init from the v4 checkpoint above, 4000 envs, 150 it.
  It checks force scaling, stage logic, saturation and NaNs at scale.
- Result: 150 it, no NaN; assist force 25–32 N mean; stage r_u throughout (|S1| ≤ 2 %);
  torque p99 5.7–8.2 Nm, saturation 0.15–1.1 %; return −28 → −13; episodes end at
  the 10 s lying rule. Valid. Proceed.

## ftsr_repro_v1 (Phase D, faithful FTSR reproduction)

- Date: 2026-10-07 03:12. Code `8be792b` (tree: docs only modified).
- Task `Mjlab-FTSR-MiniPi`, init `/home/huy/minipi_getup/logs/rsl_rl/minipi_ftsr/2026-10-07_02-51-08_ftsr_pretrain_rw_v4/model_300.pt`, 4000 envs (3000 teacher / 1000 student),
  seed 42, 8000 it, assist off from it 3000, β = 0.001, full DR, multi-pose resets.
- Progress (training log):
  - Stage transitions: r_u → r_s at it 761, r_s → r_w at it 815.
  - it 1000: |S1| 0.94, |S2| 0.92, every training episode stands with assist, torque
    p99 6.7 Nm, saturation 0.4 %.
- Eval without assist (3 seeds × 400 envs): it 500 and it 1000 both 0 % success; the
  torso never rises above the drop height. Diagnostic eval of it 1000 with the assist
  (schedule point 1000, stage starting at r_u): 16.5 % stood, supine 46 %. **At 1000
  the policy fully depends on the assist**, as expected while t_coeff = 0.67.
- Eval without assist (3 seeds × 400 envs, all four poses, `logs/ftsr_analysis/results.csv`):

  | it | success | supine / prone / left / right | t_stand | lin err (moving) | yaw err | cmd success | rec. torque p99 / sat | walk torque p99 |
  |---|---|---|---|---|---|---|---|---|
  | 500 | 0 % | 0 / 0 / 0 / 0 | – | – | – | 0 | 4.3 / 0.1 % | – |
  | 1000 | 0 % | 0 / 0 / 0 / 0 | – | – | – | 0 | 3.5 / 0.1 % | – |
  | 1500 | 0 % | 0 / 0 / 0 / 0 | – | – | – | 0 | 8.0 / 0.9 % | – |
  | 2000 | 1.3 % | 0 / 2 / – / – | 3.9 s | – | – | – | 8.5 / 1.6 % | – |
  | 2500 | 77.2 % | 50 / 90 / 82 / 87 | 2.1 s | 0.075 | 0.24 | 48 % | 8.5 / 1.5 % | 7.7 |
  | 3000 | 99.7 % | 99.7 / 99.7 / 100 / 99.3 | 0.99 s | 0.048 | 0.19 | 74 % | 8.4 / 1.5 % | 7.8 |

- Assist force and torque are exactly 0 from it 3000 (TensorBoard `Assist/force_max`,
  `Assist/time_coeff`).
- **Bug found at 3000** (mine): `FTSR/constraint_active` stayed 1 after t_tag. The
  time-out bootstrap adds `gamma·V_C` to the stored costs, so the standardized
  cost-critic residual (std 0.18 of the advantage) kept being injected. Fixed in
  `75bbbce` by gating on the raw rollout costs. The run was stopped at it 3111.
  Checkpoints 3050/3100 in this dir include ~110 updates with that noise and are not
  used. `model_3000.pt` predates any effect.

## ftsr_repro_v1_cont3000 (Phase D continued, 3000 → 8000)

- Date: 2026-10-07 04:05. Code `75bbbce`. `--agent.resume` from
  `ftsr_repro_v1/model_3000.pt`: weights, optimizer, LR, iteration 3000 and env
  step counter (assist stays 0) restored. 5000 more iterations, same everything
  else. Together with ftsr_repro_v1 (0–3000) this is the full 8000-it reproduction.
- Peak motion during the get-up, compared with the previous high-torque baseline
  (`ftsr_stand_v4/model_3850`, measured with the same substep monitor, supine only,
  `logs/ftsr_analysis/baseline_ftsr_stand_v4.json`):

  | | baseline stand_v4 | FTSR it 3000 |
  |---|---|---|
  | peak joint speed mean / max (rad/s) | 15.7 / 22.6 | 17.1 / 23.6 |
  | peak base vz mean / max (m/s) | 1.65 / 2.39 | 1.46 / 2.25 |
  | peak roll/pitch rate mean (rad/s) | 8.3 | 13.3 |
  | torque | hits 16 Nm | capped at 9 Nm, 1.8 % ≥ 8.1 Nm |
  | time to stand | 0.77 s | 1.01 s |

  The faithful FTSR get-up is as violent as the baseline, only inside the 9 Nm
  envelope. Smoothness isn't met.

## ftsr_tune_smooth_v2 (tuning round 1)

- Date: 2026-10-07 04:20. Code `1dfa14f`. Task `Mjlab-FTSR-MiniPi-TuneSmooth1`.
  Init: pretrain v4 model_300. 4000 envs, seed 42, 4500 it (assist ends at 3000, then
  1500 assist-free). Runs in parallel with the continuation on the same GPU.
- Hypothesis: the jerk comes from r_u/r_s having no speed cost. Change (one family,
  recovery motion regularization): dof_vel r_u/r_s 0 → −0.05 (r_w unchanged at
  −0.01); soft torque-limit penalty above 7.2 Nm 0 → −2 in all stages.
- Expected: peak joint speed and roll/pitch rate well down (target < 10 rad/s),
  saturation down, time to stand up (~2 s), success still > 95 %.
- Final results (eval without assist, 3 seeds × 400 envs, `logs/ftsr_analysis/results.csv`):

  | it | success | sup / pro / left / right | t_stand | lin err (moving) | yaw | cmd success | rec. τ p99 / sat≥8.1 | walk τ p99 / sat | peak q̇ | peak vz | peak ω_xy |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 3500 | 99.8 % | 99.3 / 100 / 100 / 100 | 0.91 s | 0.048 | 0.17 | 78 % | 8.4 / 1.4 % | 8.0 / 0.9 % | 17.7 | 1.48 | 13.8 |
  | 4000 | 99.6 % | 99.7 / 99.7 / 99.3 / 99.7 | 0.94 s | 0.046 | 0.16 | 79 % | 8.4 / 1.4 % | 8.0 / 0.8 % | 17.8 | 1.49 | 14.2 |
  | 5000 | 99.5 % | 100 / 99.0 / 100 / 99.0 | 0.80 s | 0.044 | 0.16 | 79 % | 8.4 / 1.3 % | 8.1 / 0.9 % | 17.1 | 1.39 | 13.8 |
  | 6000 | 99.7 % | 100 / 99.0 / 99.7 / 100 | 0.75 s | 0.046 | 0.16 | 79 % | 8.3 / 1.3 % | 8.0 / 0.9 % | 16.9 | 1.38 | 13.9 |
  | 7000 | 99.7 % | 99.7 each | 0.72 s | 0.046 | 0.16 | 79 % | 8.3 / 1.2 % | 8.1 / 1.0 % | 17.3 | 1.39 | 14.7 |
  | 8000 | 99.8 % | 100 / 99.3 / 100 / 100 | 0.76 s | 0.047 | 0.16 | 80 % | 8.3 / 1.2 % | 8.1 / 1.0 % | 18.3 | 1.34 | 14.6 |

  Policy std grew monotonically from 0.18 (pretrained) to 3.18 at it 8000 (entropy
  bonus): ~0.8 rad of target noise per step during training.
- **Outcome: faithful reproduction succeeded on recovery and locomotion**: paper-like
  timing (independent recovery appears 2000–2500, ~100 % at 3000, stable to
  8000). **It fails the smoothness/safety criterion**: the get-up is a ~0.8 s leap
  with peak joint speeds of 17–18 rad/s, and even walking runs at a torque p99 of
  ~8 Nm against the 9 Nm cap. Kept as the reference; not acceptable as the
  deployable best.
- Why: from it 815 the population is in r_w, so every get-up after that is shaped by
  the walking reward set (stage weights are global, not per env). Lying under r_w costs
  ≈ 15/s (orientation −10 |g_xy| plus the lost height reward), and r_w's only speed
  cost (−0.01 Σq̇²) is far too weak to slow a sub-second get-up.

## ftsr_tune_smooth_v2 result (tuning round 1)

- Training: never left stage r_u; return ≈ 2, stood 0 % at 4500.
- Eval without assist: it 3000 and 4500 → **0 % success** (peak q̇ still 11.7 rad/s:
  flailing on the ground).
- **Outcome: rejected.** A speed cost of −0.05 Σq̇² in r_u/r_s from the start makes
  staying on the ground preferable before the get-up skill is ever found. It is the
  same over-constraint failure as the old `safe_v*` runs, now with soft costs. Also,
  even if it had worked, r_u/r_s stop shaping the get-up once the population
  reaches r_w.

## ftsr_tune_smooth_v3 (tuning round 2)

- Date: 2026-10-07 13:08 (after a machine reboot; previous runs had completed).
  Code `c632134`. Task `Mjlab-FTSR-MiniPi-TuneSmooth2`.
- Init: `--agent.resume` from `ftsr_repro_v1_cont3000/model_4000.pt` (iteration,
  step counter (assist off), optimizer restored). 2000 it, 4000 → 6000. Control:
  faithful `model_6000` (same budget).
- Hypothesis: the get-up is shaped by r_w (population in r_w since it 815), whose
  speed cost is too weak against ~15/s for lying. Change (one family, r_w motion
  regularization): r_w dof_vel −0.01 → −0.04; r_w soft torque limit (> 7.2 Nm)
  0 → −1.0.
- Expected: peak joint speed / roll-pitch rate and the torque p99 (recovery and walking)
  down, time to stand up, success ≥ 98 %, walking tracking roughly unchanged.
- **Outcome: stopped at it ≈ 5246, not evaluated.** The user re-prioritized
  (2026-10-07 ~13:30): optimize the get-up first, because a sub-second get-up can't be
  tried on hardware. r_w motion regularization targets walking, so the run was dropped.

## Play viewer fix (`4972a6a`)

- `uv run play ... --viewer viser` crashed building the velocity joystick GUI: viser's
  "Max lin_vel_y" slider needs an initial value in [0.1, 10], and the FTSR command
  keeps lin_vel_y at (0, 0). `FtsrVelocityCommand` widens only the GUI limits; sampling
  is unchanged.
- Observation from the viewer: after getting up the robot looked unstable. Measured
  with 400 envs: under a zero command it stands still (rms ω_xy 0.03 rad/s, tilt 1°,
  no stepping). The visible motion is walking under the random play commands (90 % of
  envs): it shuffles without lifting the feet (feet airborne 0–5 %), rocking at rms
  ω_xy 0.5–1 rad/s, and barely turns. Locomotion quality remains a known weakness.

## Get-up-first evaluation protocol

- `evaluate.py --commands stand`: every env gets the zero command (column `commands`).
  Baseline faithful `ftsr_repro_v1_cont3000/model_6000.pt` under it (3 seeds × 400):
  success 99.8 %, t_stand 0.73 s (p90 1.1), peak q̇ 16.9 rad/s (max 23.6), peak v_z
  1.38 m/s, peak ω_xy 13.8 rad/s, recovery τ p99 8.2 Nm (sat 1.15 %).
- Time profile (model_6000): the 0.5 s settle hold has q̇ ≤ 4 rad/s; the peaks
  (12–14 rad/s) are the policy's own motion between 0.4 and 1.5 s.

## ftsr_getup_soft_v1 (get-up-first round 1)

- Date: 2026-10-07 13:46. Code `b5051f0`. Task `Mjlab-FTSR-MiniPi-GetupSoft`.
  Resume from `ftsr_repro_v1_cont3000/model_6000.pt`, 1500 it (6000 → 7500). Two
  earlier launches in the same tmux window were stopped by a `KeyboardInterrupt` from
  the attached client (logs `train_ftsr_getup_soft_v1_interrupted*.log`), so the run
  moved to window `getup:soft`.
- Changes: zero velocity commands (`stand_only`); soft speed caps, squared excess over
  joint speed 4 rad/s (−0.5), torso v_z 0.3 m/s (−50), roll/pitch rate 1.5 rad/s
  (−1.0), all stages, ramped in over 300 it.
- Hypothesis: in r_w each second upright earns ≈ 16 (height + zero-command tracking),
  so rising early always pays; caps sized to cost more than that for the faithful
  peaks would slow the get-up.
- Results (stand eval, 3 seeds × 400):

  | it | success | t_stand mean / p90 | peak q̇ | peak v_z | peak ω_xy | τ p99 |
  |---|---|---|---|---|---|---|
  | 6000 (base) | 99.8 % | 0.73 / 1.10 s | 16.9 | 1.38 | 13.8 | 8.2 |
  | 6500 | 99.3 % | 0.91 / 1.44 s | 16.9 | 1.34 | 11.4 | 8.2 |
  | 7000 | 99.1 % | 1.30 / 2.04 s | 15.8 | 1.22 | 10.1 | 8.0 |
  | 7200 | 98.9 % | 1.25 / 1.88 s | 15.9 | 1.22 | 9.9 | 7.9 |
  | 7250 | 63.3 % | 1.23 / 1.76 s | 14.1 | 1.05 | 7.4 | 5.4 |
  | 7500 | 58.6 % (prone 2.7 %) | 1.19 / 1.58 s | 13.4 | 1.00 | 5.9 | 5.0 |

- Collapse at it ≈ 7250: training return 242 → −50 within 40 it, prone get-up lost,
  stage manager fell back to r_s/r_u by 7480. KL spiked to 0.035/0.048 (target 0.01)
  while the adaptive LR was already ~1e-5–1e-4. The action noise std had grown
  steadily from 2.7 to 3.15 under the 0.01 entropy bonus (with a 0.25 action scale,
  ±0.8 rad of target noise), so rollouts were very noisy.
- **Outcome: partial.** Speed caps slow the get-up (t_stand +70 %, ω_xy −28 %) but peak
  joint speed stays ~16 rad/s, far from a hardware-friendly get-up, and the run is
  unstable past 7200. Best v1 checkpoint: model_7200 (model_7000 equivalent).

## ftsr_getup_soft_v2 (get-up-first round 2)

- Date: 2026-10-07 ~14:30. Code `6e44234`. Task `Mjlab-FTSR-MiniPi-GetupSoft2`.
  Resume from `ftsr_getup_soft_v1/model_7200.pt` (last checkpoint before the
  collapse), 1500 it (7200 → 8700), window `getup:soft`.
- Changes vs v1: (a) reward, rise schedule: torso height above smoothstep(0.10 →
  0.345 m over 2.5 s after the settle hold) + 3 cm costs −100/s per metre (ramped over
  300 it); joint speed cap weight −0.5 → −1.5. (b) Stability: `entropy_coef` 0.01 → 0
  (CLI), so the action noise stops growing (std 3.08 at 7200).
  Two families at once, knowingly: (b) alone doesn't slow the get-up, and v1 showed
  (a) on its own risks the same collapse.
- Expected: t_stand ≥ 2 s, peak q̇ clearly below v1's ~16 rad/s, success ≥ 95 %, no
  collapse.
- Results (stand eval, 3 seeds × 400):

  | it | success | t_stand mean / p90 | peak q̇ | peak v_z | peak ω_xy | τ p99 |
  |---|---|---|---|---|---|---|
  | 7300 | 98.3 % | 1.21 / 1.90 s | 16.0 | 1.23 | 10.1 | 8.0 |
  | 7400 | 98.5 % | 1.49 / 2.34 s | 15.9 | 1.24 | 9.9 | 7.9 |

- Training collapsed again once the terms reached full weight (ramp ends ≈ 7500):
  return 277 → −109 at 7438 (episode length 538; `low_too_long` terminations), joint
  speed cost up to −4.4/s, stage manager back to r_u by 7613, stood ≈ 0.6. The noise
  std stayed ~3.03 (entropy 0 worked), so action noise growth wasn't the only cause.
- **Outcome: stopped at it 7652; rejected as a training recipe.** Best checkpoint
  model_7400 (slowest get-up with success ≥ 98 %), but peak joint speed is unchanged
  at ~16 rad/s. Reward-side speed limits on a policy that already gets up dynamically
  only stretch the slow phases; when they bind, PPO falls off the get-up instead of
  finding a gentler motion.

## ftsr_getup_gentle_v1 (get-up-first round 3)

- Date: 2026-10-07 ~15:05. Code `6bcf2f2`. Task `Mjlab-FTSR-MiniPi-GetupGentle`.
  Init (weights only) `ftsr_pretrain_rw_v4/model_300.pt`, as the faithful run;
  4000 it, assist schedule as faithful (off at 3000). Window `getup:soft`.
- User decision (2026-10-07): torque envelope for the get-up raised to 12-13 Nm;
  12.5 Nm used (DR motor strength 0.85-1.05 → ≤ 13.1 Nm). Faithful task stays 9 Nm.
- Changes vs faithful: joint target rate limit 0.1 rad per 20 ms step (5 rad/s, hard,
  in the action term; must be applied by the deployment wrapper too); 12.5 Nm; zero
  velocity commands. Rewards: faithful stage table (no speed penalties).
- Hypothesis: a hard target rate limit rules out the jump the reward penalties couldn't
  remove, without the reward-side instability; the extra torque keeps a slow get-up
  feasible.
- Check (64 envs, random actions): actuator forcerange ±12.5, max target step
  0.1000 rad, max |τ| 12.6 Nm.
- Evaluator bug found at it 1000: `evaluate.py` always built the faithful env
  (9 Nm, no rate limit), so the first GetupGentle evaluations (600, 1000) ran the
  policy in the wrong env. Fixed with `--env-task` (column `env_task`); the two
  invalid rows were removed from results.csv. GetupSoft v1/v2 evaluations are
  unaffected (they differ from the faithful env only in rewards and commands, and the
  evaluator sets the commands itself).
- Peak metrics now start after the settle hold (the reset drop alone gives torso
  v_z ≈ 0.96 m/s). Re-check of faithful model_6000: unchanged (q̇ 16.9, v_z 1.37,
  ω_xy 13.7; success 99.9 %).
- Correct-env eval, no assist: it 600 and 1000 → 0 % (faithful was 0 % at 1000 as
  well); peaks while trying q̇ 5.7–5.9 rad/s, ω_xy 3.9 rad/s.
