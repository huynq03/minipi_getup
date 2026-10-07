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
