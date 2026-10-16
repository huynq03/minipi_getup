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
scripts/play.sh [CHECKPOINT] [-n robots] [-p port]      # viewer; default checkpoint model_1100, its action scale
scripts/play_export_onnx.sh [CHECKPOINT] [...]          # export <ckpt dir>/onnx/<stem>.onnx (+ .json), then play
```

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

## After model_1100: prone get-up only, no sliding, action scale 0.40

> **Code state (2026-10-16):** the training code is back at commit `2050c04` (rev13, action-scale floor
> 0.25). That commit trained `getup_2600` (ONNX md5 `ffcc3285…`), which ran best on the robot at scale
> 0.4. The experiments below are kept in `git stash` ("wip 2026-10-15 …"): prone-only, floor 0.40,
> friction, rise no-slip, knee 0.10, and an EMA change that collapses training. Their checkpoints and
> ONNX files (`model_4000`, ft2600) were trained with that stashed code. `export_onnx.py` and `scripts/`
> stay.

Training from scratch on 2026-10-15 (run `rev13_prone_only_scratch`, 1024 envs, 8000 iterations):

- **Prone resets only** (`safe_recovery.reset_classes = ("prone",)`), plus the 15 % stance starts. The
  goal is model_1100's knee-supported prone get-up at the lower scale. `evaluate --classes` picks the
  evaluated falls (default: all five).
- **`style_rise_foot_slip = -12`**: loaded-foot translation cost while rising (`rise_gate`: base above
  ~0.10 m, not yet standing), on top of `loaded_sole_slip` (-6) and `loaded_foot_translation` (-4).
- **`action_scale_floor = 0.40`**: the intended deployment scale.
- **Resumed at iteration 2800** (run `rev13_prone_only_r2800`, +2500 iterations), after a plateau from
  ~2000. The velocity termination had settled at ~10.4 rad/s and cut 26 % of episodes. That held
  training success at ~0.69 and the penalty curriculum at 0.22 strength. model_2200 (evaluated unassisted,
  prone) was faster than model_1100 (1.06 s p50) but slid more while rising (35 vs 24 cm), stood on a
  straight right knee every time, and peaked at 11 rad/s. Changes: `velocity_termination = False`,
  `style_joint_overspeed` -2 → -4, `style_standing_knee_extension` -16 → -48.
- **Result of the resume (to 5300):** training success 0.97, but the targets did not move. Rise sliding
  stayed at ~33 cm and joint speed rose to ~10.8 rad/s p95. The knee settled just past the -0.25
  deadband, where the cost is ~0. The ~+60/s task reward dwarfs the -0.7/s slip cost. Best checkpoint:
  `model_4000` (copied with its ONNX to `checkpoints/rev13_prone_only/`).
- **Fine-tune of model_1100** (run `rev13_m1100_prone_s040`, from 1100 to 2600): prone resets, action-scale
  floor 0.40, `style_rise_foot_slip = -12`, everything else as rev13 (velocity termination on, knee -16,
  overspeed -2).
- **Fine-tune result (`checkpoints/rev13_m1100_finetune/model_2600.pt` + ONNX, scale 0.40):** prone,
  unassisted: 1.28 / 1.40 s to stand, 23 cm rise sliding, joint speed 7.0 / 8.0 rad/s p50 / p95
  (model_1100: 8.4 / 9.8). The knee ended straighter (calf -0.26). Overspeed terminations (~23 %)
  held the penalty curriculum at 0.52.
- **Fine-tune of ft2600** (run `rev13_ft2600_fric_noslip`, 2600 to 3600):
  - Friction 0.8-1.2 effective (`domain_rand.friction_range = [0.85, 1.65]`, evaluation 1.0 via
    `nominal_friction = 1.25`).
  - `task_rise_no_slip`: exp(-0.5 x loaded-foot translation x rise_gate) multiplies the task product.
  - Penalty EMA ignores overspeed terminations.
  - Knee deadband 0.25 -> 0.10.
  - Friction alone barely helps: ft2600 at 1.0 still slides 21 cm (35 cm p95). The sliding is
    commanded, not a lack of grip.
- **Hip roll free.** A run that penalised outward hip roll above 0.8 rad while rising (-40) kept the hips
  closed while kneeling (the gate is already 27-73 % open at 0.07-0.12 m). Without a wide leg to brace
  on, prone get-ups fell forward and retried: 2.40 s p50 / 4.46 s p95 at iteration 1400 versus 1.26 /
  1.45 s for model_1100.
- **Prone first from every fall (dropped).** Zero task reward until the torso had been chest-down, with a
  chest-down score as the only target reward. In 400 iterations supine starts never rolled over: the
  chest-down fraction stayed at ~0.6 and mean reward flat at ~-650, while the pull force still lifted
  ~17 % of episodes upright without the route.

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
