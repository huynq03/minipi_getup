# Training trend snapshot — 2026-10-13 23:46 ICT

Run: `20261013_234113_clpai_v2_contact_tmux`, tmux `clpai_v2_train`.
Snapshot at iteration 293/8000. Metrics come from TensorBoard, comparing first
50 iterations of the initial run against the last 30 of the resumed run.
Exact values are in `CL_PAI_V2_TRAINING_TREND.json`.

| Metric | First 50 mean | Current last 30 mean |
|---|---:|---:|
| Task reward | 0.0473 | 0.1026 |
| Torso near COM reward | 0.0224 | 0.0342 |
| Sole coverage reward | 0.00138 | 0.00841 |
| Feet separation penalty | -1.607 | -0.0130 |
| Contact slip penalty | -0.2119 | -0.2007 |
| Mean absolute joint velocity, rad/s | 3.638 | 2.007 |
| Base height, m | 0.113 | 0.138 |
| Soft joint-limit penalty | -0.692 | -1.141 |
| Fallen-start success | 0.00077 | 0 |
| Standing both soles coverage >=90% fraction | 0 | 0 |
| Episodes with actuated speed above evaluation ceiling | 99.34% | 99.98% |

Interpretation: feet-separation violations have fallen sharply and average motion
has slowed. Upright preparation/torso-COM and sole coverage rewards are improving,
but the policy has not learned reliable get-up: recent fallen-start success and
standing two-sole >=90% coverage remain zero. Slip is still substantial, and
soft joint-limit penalties have worsened, suggesting more time near/outside soft
position limits. These are training-rollout statistics with exploration and
assistance, not deterministic unassisted evaluation or proof of stable sitting.

At this snapshot, mean assistance is 21.81 N, mean action scale remains 1.0 and
feet-supported success EMA is 0. No automatic reward/controller changes were
made in response to this status query; the run remains active. Compare later
checkpoints with deterministic force=0 evaluation before judging deployment quality.
