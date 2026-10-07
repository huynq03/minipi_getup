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
