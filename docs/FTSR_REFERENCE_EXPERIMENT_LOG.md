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
