# HoST staged recovery on Mini-Pi (revision 13)

The HoST get-up policy for the HighTorque Mini-Pi 12-DOF biped, trained in mjlab. This is the one
line of work with a working result: `checkpoints/rev13/model_1100.pt` stands up from every fall
orientation without assistance.

## Code and asset

| Path | Role |
|---|---|
| `src/minipi_getup/host/`, `host_rl/` | HoST port to mjlab (env, PPO with reward groups) |
| `src/minipi_getup/host_safe_recovery/` | Base recovery env: contact sensors, safety and posture terms |
| `src/minipi_getup/host_staged_recovery/` | Revision 13: staged rise, planted soles, stance; `REWARDS.md` lists every term |
| `.../xmls/cl_pai_host_torso_host_legs_hip120_legcoll.xml` | Robot asset (SHA256 `ea7d909d…`, pinned in `asset.py`): HoST primitive collisions, 120° hip roll |
| `checkpoints/rev13/model_1100.pt` | Reference policy (run `20261013_123752_rev6a`, commit `14861b3`) |

## Setup

```bash
uv sync                      # Python 3.13, torch cu126, mjlab 1.6.0, warp 1.18 (cu12)
source .venv/bin/activate
export MUJOCO_GL=egl WARP_CACHE_PATH=$PWD/.warp-cache-cu12
```

Verified on Ubuntu 22.04, RTX 3070 Ti 8 GB, driver 580. 1024 envs fit in 8 GB; 4096 need about 10 GB.

## Train, resume, watch, evaluate

```bash
# from scratch (refuses uncommitted package changes unless --allow-dirty)
python -m minipi_getup.host_staged_recovery.train --num-envs 1024 --max-iterations 8000 --run-name NAME
# resume: policy, optimizer and curriculum state; --max-iterations is added to the checkpoint's
python -m minipi_getup.host_staged_recovery.train --resume checkpoints/rev13/model_1100.pt --max-iterations 1000
# viewer at http://localhost:8080 (pull force, DR and training terminations off)
python -m minipi_getup.host_staged_recovery.play_viser --checkpoint checkpoints/rev13/model_1100.pt --action-scale 0.466
# unassisted evaluation per fall orientation, JSON report (+ optional trajectories)
python -m minipi_getup.host_staged_recovery.evaluate --checkpoint checkpoints/rev13/model_1100.pt \
  --action-scale 0.466 --num-envs 250 --save-traj 40 --out eval/model_1100.json
```

Runs go to `logs/host_staged_recovery/<stamp>_<name>/`: `model_<iter>.pt` every 100 iterations,
TensorBoard events, and `config.json` (commit, diff, asset hashes, reward scales). A checkpoint resumes
only with the same variant and asset hash. Use the training action scale for play and evaluation
(`Episode/action_scale` at that iteration; 0.466 for model_1100).

Useful TensorBoard scalars: `Episode_end/success`, `rise_two_foot_frac`, `penalty_foot_success_ema`
(penalty curriculum, starts at 0.60, full at 0.85), `Curriculum/force_mean`, `Episode/action_scale`
(floor 0.30), `standing_*`, `post_stand_drag_m`.

## Results: model_1100

Evaluated on 2026-10-15 with the training code and asset: 250 episodes, 10 s, no pull force, no DR, no
observation noise, action scale 0.466.

| Metric | Value |
|---|---|
| Success, all five fall classes (supine, prone, left, right, intermediate) | 100 % |
| Time to stand after actuation | 1.22 s median, 1.65 s p95, 2.58 s max |
| Feet loaded / rise onset (base 0.24 m) | 0.46 s / 0.98 s |
| Stance width at rise onset / final | 0.32 m / 0.137 m |
| Final posture RMS, sole tilt | 0.16 rad, 2.7° |
| Knee ground contact | 0.25 s mean (prone starts 0.49 s) |
| Peak joint speed | 12.7 rad/s p95, 15.6 max (hip roll); every episode exceeds the 6 rad/s hardware limit |
| Outward hip roll while rising (base 0.12–0.30 m) | 1.63 rad median peak |
| Grounded-sole sliding before / after standing | 18.4 cm per episode at ~0.30 m/s / 7.2 cm at 0.004 m/s |

How it gets up, from 40 recorded episodes:

- **Supine:** it sits up directly; it never turns chest-down. While the torso comes upright it opens
  both hip rolls to ~1.5 rad (frog squat) with the soles planted beside the hips, then stands and
  closes the legs. Closing the loaded legs is where the soles slide.
- **Prone:** it flexes the hips (+1.2) and knees (calf ~1.47), pushes the torso back to upright, and
  continues through the same wide squat.
- **Side and intermediate:** it rotates upright on wide hip roll (1.1–1.6 rad); 43 % pass briefly
  through chest-down.

## After model_1100: narrower rise without sliding

Two terms are active only while rising (`rise_gate`: base above ~0.10 m, not yet standing):

- `style_rise_hip_roll = -40`: squared outward hip roll above 0.8 rad.
- `style_rise_foot_slip = -12`: loaded-foot translation (the `loaded_foot_translation` cost) on top of
  the existing `loaded_sole_slip` (-6) and `loaded_foot_translation` (-4).

Training from scratch with them started on 2026-10-15 (1024 envs, 8000 iterations, ~7.5 h).
Compare it against the table above.

## Discarded attempts (code and logs removed)

- **Fitted convex self-collision shells (`cl_pai_v2.xml`):** the shells blocked hip pitch. The policy
  then pressed joints into each other at the 16 Nm limit and never stood unassisted.
- **Prescribed prone route (`host_rev6_recovery`):** chest-down → COM over the soles → rise. It
  reached a supported crouch, ~90 % unassisted, after the pull force was tied to that stage. It
  never rose from it in 8 runs. model_1100 shows the direct sit-up with wide hip roll works better.
- **Training failures worth remembering:**
  - Stage goals must be reachable with joints inside about 20–80 % of their range. One was not, and
    the policy locked at the limits: tanh outputs at ±1, the adaptive LR at its 1e-2 ceiling.
  - Joint-limit costs in the `regu` group (weight 0.1) are too weak to prevent this.
  - A pull force that only shrinks once the robot stands never shrinks.
