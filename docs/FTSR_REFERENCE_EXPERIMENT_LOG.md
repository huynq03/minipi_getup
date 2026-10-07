# FTSR reference rebuild: experiment log (append-only)

## 2026-10-07: state frozen, audits written, blocked on motor data

- Archived the experimental tree: tag and branch `archive/ftsr-experiments-20261007`
  at `0a8cb71` (= `main` = `origin/main`). Working tree was clean. Nothing pushed.
- New branch `rebuild/ftsr-reference-minipi` from `8a2a990` (pre-FTSR baseline). Outside
  `src/minipi_getup/ftsr/` and `docs/`, `8a2a990..0a8cb71` differs only by the one-line
  `import minipi_getup.ftsr.config` in `src/minipi_getup/__init__.py`: the Mini-Pi
  asset and control files are identical.
- Audits: `docs/FTSR_REFERENCE_AUDIT_V2.md`, `docs/MINIPI_PHYSICAL_MODEL_V2.md`,
  `docs/MINIPI_DEPLOYMENT_CONTRACT_V2.md`.
- Key findings: the release's force-guided optimization is a no-op (nothing writes
  `extras["force"]`); teacher-identity shuffle bug; history stack holds 3 distinct
  frames; one-step latent lag; single reset pose; no termination; no checkpointed env
  counter; no walking pretrain.
- **Blocker (stop condition "hardware motor data is contradictory")**: installed motors
  are HTDW-5047-36-NE per the loaded SDK config, the speed numbers in hand are for
  HTDW-5036-02 and come from an unverified summary (7.85 rad/s), the vendor URDF says
  21 rad/s, and torque caps disagree (16 / 6 ankle roll / 35 / none). The plant must be
  final from pretraining iteration 0, so no plant, pretraining or training was started.

## 2026-10-07: audit corrections after review (no code, no training)

- Eq. 5–8: separate cost critic no longer stated as required; `J_t,Ci` estimator,
  `A_Ci` estimator, standardization scope, cost units and per-group terms marked
  `PAPER_AMBIGUITY`; normalized costs are a hypothesis; β = 0.001 kept as the paper
  baseline, its meaning depends on cost units.
- Critic: release final critic input is `[E_t(x^t), o_t*, height map, base height]`
  = 12 + 222 = 234 with the teacher latent for all rows; paper Table I says
  `[z_t, s_t]`, s ∈ R^188, and Fig. 2 routes the student latent for student agents.
  Both recorded as paper-vs-release conflicts.
- h2 = 0.259 withdrawn: the release g4 condition depends on height and time; h1/h2/h3
  stay candidate parameters.
- Motor evidence reclassified by layer (physical capability / firmware / controller
  software / simulation); URDF 21 Nm / 21 rad/s are URDF limits, not motor data.
  Blocker unchanged.
- `T_max`, `μ` marked UNRESOLVED calibration parameters.
- New invariant: `q_target` clipped to physical per-joint ranges identically in
  training and `deploy.yaml`.
- 500 Hz sim PD vs 1 kHz-refreshed firmware PD recorded as an approximation needing
  sensitivity validation.

## 2026-10-07: operational joint-speed contract added (user requirement; audits only)

- New hardware/deployment contract for initial real get-up tests: 50 Hz policy,
  `q_target` slew 3 rad/s = 0.06 rad/step relative to the previous commanded target,
  applied after raw clip → absolute target → physical per-joint clip; measured q̇ never
  clamped; soft q̇ envelope ≈ 3 rad/s (moderate penalty/monitor), emergency ≈ 4 rad/s.
- Replaces the earlier "no slew limiter" conclusion. mini_pi_fsm has no slew limiter:
  a GetUp-state runtime slew (with entry initialization and a state-local 4 rad/s
  fault) is documented as required before real actuation; not implemented.
- Motor no-load speed is separated from operational speed. Simulation must be evaluated
  under H-conservative (ω₀ 7.85 rad/s) and H-loose (ω₀ 21 rad/s), both with τ_stall
  21 Nm and τ_cap 16 Nm as hypotheses; neither is the verified motor speed;
  H-conservative success is preferred.

## 2026-10-07: Phase A, implementation and static validation (commit bcd00ee)

- Audit 9791755 accepted by the user; H-conservative (ω0 7.85, stall 21, cap 16) is
  the nominal provisional plant, H-loose (ω0 21) evaluation only.
- Package `src/minipi_getup/ftsr_ref/`, tasks `Mjlab-FTSR-Ref-MiniPi-Walk` /
  `-Recovery`. Decisions and classes: `docs/FTSR_REF_IMPLEMENTATION.md`.
- Finding: with the motor envelope and no armature, 2 ms physics lets light joints
  exceed the no-load speed (ankle roll 71, ankle pitch 18, hip yaw 13 rad/s vs 7.85);
  the explicit per-step forcerange oscillates on the ankle roll (I = 4.4e-4 kg m²).
  Fix without inventing a parameter: implicit piecewise actuator law + 0.5 ms × 40
  physics (calf/yaw/ankle pitch then saturate at 7.83–7.94 rad/s). Armature remains an
  open physical uncertainty.
- Finding: passive settling of the fallen resets takes ~2 s under deploy Passive
  (kd 1); window set to 100 steps (release 0.6 s).
- Finding: Eq. 4 with F_max = 1.474 m g lifts a passive lying robot to 0.32 m within
  1 s as the stage advances; kept as derived, 1.0 m g ablation planned.
- validate.py: 25/25 pass. Smoke train: 4.2 s/iteration at 4000 envs.

## 2026-10-07: Phase B, walking pretrain v1 started

- `uv run train Mjlab-FTSR-Ref-MiniPi-Walk --env.scene.num-envs 4000
  --agent.max-iterations 600 --agent.run-name ftsr_ref_walk_v1` (tmux getup:ref,
  log `logs/ftsr_ref_walk_v1.log`). No gait clock, no legged shaping (release r_w with
  robot replacements only), slew 0.06 rad/step, H-conservative.

## 2026-10-07: walking pretrain v1 finished; performance refactor (no method change)

- `ftsr_ref_walk_v1` completed 600 iterations (`logs/rsl_rl/minipi_ftsr_ref_walk/
  2026-10-07_19-22-*_ftsr_ref_walk_v1`). Evaluation below.
- Performance refactor of the training hot path. Files: `ftsr_ref/mdp/actions.py`,
  `ftsr_ref/rl/runner.py`, `ftsr_ref/mdp/stages.py`, `ftsr_ref/mdp/rewards.py`.
  - Training monitor: per-physics-step `torch.bincount` histograms (torque, qdot,
    tracking error) replaced by per-joint device accumulators (count, sum/max of
    |tau|, |qdot|, |q* - q|, counts |qdot| > 3 / > 4, |tau| > rated, |tau| >= 0.95 cap,
    slew saturation). Training logs now report mean / max / fractions, not p95/p99
    (those stay in `evaluate.py`, unchanged).
  - GPU->CPU syncs removed: slew counters (`float(act.sum())`, `float(sat.sum())`
    every policy step), `bool(self.force.any())` every physics step (now a host-side
    `_assist_live` flag; one zero-wrench write at t_tag), `float()` of every rollout
    log value and of `log_assist()` every policy step (device sums, one reduction
    per iteration), `bool(cost.max() > 0)` every step (device flag, read once),
    per-step `nonzero()` + `tolist()` episode bookkeeping (device [T, N] buffers, one
    `tolist()` per iteration, same order), PPO / student loss stats per minibatch,
    stage fractions (none for a fixed stage; one read instead of two per step for the
    dynamic stage), masked assignment in `pen_max_velocity` (now `torch.where`).
  - Per-physics-step `zeros_like` / `full_like` temporaries removed; passive-window
    gains computed once per policy step.
  - Remaining syncs: dynamic stage decision (1 per env step, host-side reward
    weights), adaptive-LR KL comparison (1 per minibatch, release algorithm), mjlab
    internals (reset / termination indexing).
- Unchanged: PHYSICS_DT 0.0005, DECIMATION 40, plant, action contract, rewards,
  observations, PPO / Eq. 5-8 / student MSE, teacher-student split. validate.py 25/25.
- Benchmark (same command before/after; baseline = b3b273a via stash):
  `uv run train Mjlab-FTSR-Ref-MiniPi-Walk --env.scene.num-envs 4000
  --agent.max-iterations 20 --agent.run-name perf_bench_{base,opt}`; medians over
  iterations 5-18 (first 5 and the final save iteration excluded); tc/tl from the
  console (0.1 s resolution), total/FPS from TensorBoard `Perf/total_fps`:

  | | collection | learning | total | FPS | GPU util | GPU mem |
  |---|---|---|---|---|---|---|
  | before | 3.8 s | 0.2 s | 3.946 s | 24 329 | 90 % | 1770 MiB |
  | after | 3.2 s | 0.2 s | 3.365 s | 28 527 | 97 % | 1770 MiB |

  Total −14.7 % per iteration (+17.3 % FPS), all in collection. ~3.2 s of
  collection remains, essentially the 0.5 ms x 40 physics.

## 2026-10-07: walking v1 evaluation (model_600) and continuation

Deterministic student, H-conservative, 16 envs per command, 10 s (8 s measured):

- vx: corr 0.999; -0.34 -> -0.29, -0.15 -> -0.12, 0.15 -> 0.15, 0.3 -> 0.29,
  0.54 -> 0.51 m/s. Zero command: drift 3.6 cm, no stepping. Falls 0 %.
- Yaw while walking (vx 0.2, wz +-0.44): +-0.38 rad/s (86 %), no yaw bias at wz 0.
- **Yaw in place (vx 0, wz +-0.44): no turning (-0.03 / -0.08 rad/s).**
- Stepping: 3.4-5.1 touchdowns/s/foot, air fraction 0.15-0.39 while walking.
- Torque max 13.7 Nm (calf), 0 % near the 16 Nm cap, 1.7 % above 6 Nm (longest
  0.09 s). qdot > 3: 3.3 %, > 4: 0.76 % of samples, but 78 % of episodes touch > 4
  rad/s once (hip yaw / ankle roll p99 4-5 rad/s). Joint-limit margin min -0.0185 rad.
- Diagnosis: yaw reward still rising (1.64 / 1.75 / 1.84 per s at it 400 / 500 /
  599) while vx tracking saturated; pure-yaw commands are ~10 % of samples and the
  release kernel (sigma^2 0.261) still pays 47 % for standing still under 0.44 rad/s.
- Smallest reference-consistent action: no method change; resume training from
  model_600 for 300 iterations (`--agent.resume True --agent.load-run
  2026-10-07_19-22-44_ftsr_ref_walk_v1 --agent.load-checkpoint model_600.pt`,
  run `ftsr_ref_walk_v1_cont`), then re-evaluate.

## 2026-10-07: walking init accepted (model_900); recovery v1 started

- model_900 (after +300 iterations, no method change): vx corr 0.999 (-0.34 -> -0.31,
  0.54 -> 0.52 m/s), yaw while walking +-0.38 / 0.39 rad/s for +-0.44, zero command
  drift 3.6 cm, falls 0 %, 3.2-5.5 touchdowns/s/foot, torque max 12.8 Nm, 0 % near
  cap, qdot > 4 rad/s 0.6 % of samples (86 % of episodes touch it once; p99 max 5.1
  rad/s, hip yaw / ankle roll), joint-limit margin min -0.016 rad.
- Still no turning in place (vx 0, wz +-0.44 -> -0.03 / -0.01 rad/s); 300 more
  iterations did not change it. Accepted as "elementary walking" (paper Sec. III-A):
  turning in place is not part of the recovery skill, r_w keeps training during
  recovery, and a fix would change the reference command distribution or kernel
  (a method change). Known limitation, revisit only if r_w behaviour after recovery
  requires it.
- Recovery v1: `uv run train Mjlab-FTSR-Ref-MiniPi-Recovery --env.scene.num-envs 4000
  --agent.init-checkpoint /home/huy/minipi_getup/logs/rsl_rl/minipi_ftsr_ref_walk/2026-10-07_20-31-38_ftsr_ref_walk_v1_cont/model_900.pt --agent.run-name ftsr_ref_recovery_v1` (8000
  iterations, assist to 3000, no-assist eval every 500 iterations), tmux getup:ref.

## 2026-10-07: recovery v1 terminal reward breakdown; resumed at 150

- `e43da8b`: logging only. Each iteration prints up to 2 lines of the nonzero
  `Episode_Reward/*` terms (per-second episode averages already in `extras["log"]`).
  No reward, PPO, plant, stage or assist change.
- Recovery v1 stopped right after `model_150.pt` and resumed with the identical
  configuration (`--agent.resume True --agent.load-run
  2026-10-07_20-50-29_ftsr_ref_recovery_v1 --agent.load-checkpoint model_150.pt
  --agent.max-iterations 7850`, run `ftsr_ref_recovery_v1_r150`, log
  `logs/ftsr_ref_recovery_v1_r150.log`). The checkpoint restores networks, optimizers,
  lr, iteration and env step counter (assist tc continues 0.950 -> 0.942); only
  in-flight episodes were reset.
- Iterations 0-150: stage r_u throughout; height reward 0.03 -> 1.8 /s; mean reward
  -16 -> -28 (it 40) -> -2.3; S1 0.31-0.34 early then 0.23 at 150, S2 0.12-0.15;
  F mean ~27-30 N; tau max 16 (cap) on hips/calves, > 6 Nm ~1 % of samples;
  qd > 3: 11-13 %, > 4: 1.2-1.4 %; slew saturation 0.75; ~3.2 s/iteration,
  1770 MiB GPU. No NaN.

## 2026-10-07: recovery v1 eval_500 / stopped at 1000 (stage deadlock); v2 fix

Training trend (v1, with assist):

| it | rew | height /s | S1 | S2 | F N | tc | std | qd>3 | qd>4 | slew |
|---|---|---|---|---|---|---|---|---|---|---|
| 25 | -19.8 | 1.31 | 0.37 | 0.09 | 26.5 | 0.99 | 0.32 | 0.142 | 0.031 | 0.75 |
| 100 | -10.4 | 1.84 | 0.32 | 0.13 | 29.7 | 0.97 | 0.36 | 0.130 | 0.017 | 0.78 |
| 200 | 8.0 | 1.85 | 0.13 | 0.09 | 32.1 | 0.93 | 0.40 | 0.033 | 0.004 | 0.60 |
| 300 | 12.9 | 2.12 | 0.03 | 0.01 | 35.1 | 0.90 | 0.43 | 0.018 | 0.002 | 0.47 |
| 499 | 13.2 | 2.00 | 0.01 | 0.00 | 36.1 | 0.83 | 0.51 | 0.016 | 0.001 | 0.44 |
| 999 | -2.3 | 1.86 | 0.07 | 0.00 | 30.8 | 0.67 | 0.79 | 0.051 | 0.002 | 0.51 |

Stage r_u throughout; tau max 16 (cap) every iteration but > 6 Nm only 0.1-3 %.

eval_500 (no assist, 64 envs/pose, student; teacher identical): success 0 % on all
poses, never stood. Max base height p90: supine 0.081, prone 0.116, left 0.112, right
0.112 m (lying 0.069-0.083); 0 % of time above h1. Final: supine sits at 0.05 m with
upright cos 0.65 (torso raised ~50 deg, seated on the pelvis), prone stays prone
(cos 0.11), sides 0.05-0.09 m (cos 0.3-0.5). Gentle: qd > 3 0.3-0.7 %, > 4 0-0.1 %
(but 97-100 % of episodes touch 4 rad/s once), tau max 16 on one hip per pose, near
cap 0-0.1 %, slew saturation 0.06, joint-limit margin min -0.055..-0.081 rad.

Diagnosis (deadlock):
1. r_u height reward 4 exp(-|h - h1|/0.0552) peaks exactly at the S_1 threshold h1:
   dR/dh = +71 /m just below h1 and -71 /m just above, gain 0.18 -> 0.20 m is 0. The
   optimum puts the population *at* h1, so ">2/3 strictly above h1" is never rewarded;
   standing (0.345 m) earns 0.24 vs 4.0 at h1, so the walking init's standing envs
   (S2 0.13 at it 100) were unlearned (S2 -> 0).
2. Eq. 4 is clamped to 0 at h_cmd = h1 and carries 0.5-0.6 m g just below it (36 N at
   ~0.156 m with tc 0.83); the cheapest point of the kernel's slope is to hang on the
   assistance below h1. Without assistance the policy does not lift off at all.
v1 after 500: std 0.51 -> 0.79, reward 13 -> -2, still r_u: stopped at model_1000.

Fix (one change, recovery v2): r_u height-reward target 0.19 -> 0.214 m = h1 x
0.45/0.40, the release's ratio of the height target active at its g2->g3 switch to
that switch (the release decouples target and group). REFERENCE-consistent (release
semantics; paper ties h_cmd to both; PAPER_AMBIGUITY resolved toward the release).
Unchanged: S_1/S_2 thresholds, 2/3 rule, Eq. 4 h_cmd (= h1), all weights, sigma,
PPO, std, plant, slew, physics, assist schedule, teacher/student. Landscape with
target 0.214: dR/dh = +46 /m below and +48 /m above h1, gain 0.18 -> 0.20 = +0.95,
R(0.156) 1.41 (pull 25 /m, was 39), standing 0.37. Larger targets (0.285 = 1.5 h1)
leave only 7 /m of pull at the current height, hence the smaller step.
