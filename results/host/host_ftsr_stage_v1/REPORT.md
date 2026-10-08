# host_ftsr_stage_v1 — implementation and validation

Implemented locally on 2026-10-08. **No long training was started.** This is an
FTSR-inspired HoST experiment, not a claim that direct armless recovery is solved.

## Source audit

Read-only references actually inspected:

- `/home/huy/getup_gym/getup_gym/envs/bipedal_wheeled/env.py::_reward_update`
  selects reward group 2/3/4 from population height and force scale. The tests use
  >2/3 of the population, heights 0.35/0.40 m and force scale <100. Group 1 is
  configured but not selected by this function.
- `.../bipedal_wheeled/config.py::rewards.reward_group_2/3/4`: erection uses height,
  orientation and contact; later groups increase posture/stability/motion terms.
- `.../unitree_humanoid/env.py::_reward_update` uses the same population mechanism,
  with force scale <60. Its config includes knee-down, pose and symmetry shaping.
- `.../common/reward_functions.py`: `track_base_height_exp` is exp(-absolute height
  error/sigma); `pen_base_orientation_l2` emphasizes projected gravity z+1;
  `pen_base_orientation_z_l2` detects upside-down; `rew_wheel_contact_force` matches
  summed wheel force to robot weight. `rew_knee_down` contains humanoid-specific
  angle targets. Neither wheel-force matching nor those angle targets were copied.
- `.../common/reward_manager.py::compute_reward` **adds** weighted terms times dt.
- `src/minipi_getup/host/env.py::compute_reward` **multiplies** positive task terms,
  then **adds** the other groups. `_prepare_reward_function` scales only additive
  constraints by policy dt. `host_rl/rollout_storage.py::compute_returns` normalizes
  each critic's advantage separately and combines it with group weights.

Intentional adaptation: three per-environment stages, dwell, monotonic stage latch,
continuous blending and high-water progress. No FTSR CPO, teacher/student, walking
command or population-stage switch is introduced. Existing HoST pull/beta curriculum
is preserved, including its reset-batch height decision; reward stages do not alter it.

## Files and isolation

All implementation is new under `src/minipi_getup/host_ftsr_stage/`:

- `config.py`: separate config; existing four groups and [2.5, 0.1, 1, 1] weights.
- `geometry.py`: bounded IK and collision validation of support reference poses.
- `rewards.py`: vectorized stage state, continuous shaping, success classification.
- `env.py`: HoST subclass, measurement-only foot-ground sensors and metrics.
- `train.py`: scratch/resume launcher with experiment/asset/config guard.
- `evaluate.py`: deterministic mean-action, force-OFF evaluation and metrics.
- `validate.py`: CPU logic/kinematics, GPU plant parity, PPO and baseline replay tests.

Baseline HoST, FTSR, XML, meshes and reference repositories were not edited by this
implementation. Existing uncommitted changes from earlier work were retained.
Factory/asset/torque overrides are scoped to construction and restored afterwards.

| Property | Existing real-model HoST | New stage task |
|---|---|---|
| Plant | asset-zoo `cl_pai.xml` + 9 massless marker bodies | Identical |
| Mesh/collision/inertia/limits | existing asset-zoo model | Identical |
| PD | Kp 60/40/20/60/30/10, Kd 2.4/.8/.4/2.8/1.6/.3 | Identical |
| Torque model | static clip 16 Nm; ankle roll 6 Nm; DR | Identical; no new torque-speed curve |
| Policy/PD/physics | 50/200/400 Hz | Identical |
| Action | refreshed relative target q + beta*action | Identical |
| Actor/history | 43 features x 6 = 258, 12 actions | Identical |
| PPO | four critics, existing networks, separate advantage normalization | Identical |
| Noise/DR/initial states/pull/beta | original HoST-real | Identical for training |
| Task group | positive orientation x head-height product, not dt-scaled | Identical |
| Regularization | original terms | non-joint-limit motion terms gated .1→1 late in recovery |
| Style group | old HoST style terms | stage progress + mild route/anatomy costs |
| Target group | old HoST post-task terms | supported anatomically valid standing quality/hold |
| Stage/contact/height in actor | none | None added |

The stage state and simulator-only support/height measurements affect rewards and
metrics only. Neither actor nor critics receive new privileged features. A deployed
actor keeps exactly the existing proprioceptive interface. Episode-specific stage,
hold and progress history reset with the simulated robot, also on resume; only
policy/optimizer/curriculum persist. Resuming does not pretend to resume physics.

## Frame and geometry audit

Torso `base_link` has local +Z up and +X facing upward in the existing supine reset.
The existing reset quaternion is normalized from xyzw [0,-1,0,1]. CPU MJCF forward
kinematics and the actual GPU environment agree:

| Pose | Projected gravity in torso frame |
|---|---|
| Upright | (0,0,-1) |
| Supine | (-1,0,0) |
| Prone | (+1,0,0) |
| +90 degree rotation about X | (0,-1,0) |
| -90 degree rotation about X | (0,+1,0) |

Sideways inclination = asin(abs(g_y)), **not Euler roll near the pitch singularity**.
It is zero throughout a pure sagittal sit-up. It measures 0–90 degree inclination,
not accumulated revolutions. Prone detection is g_x > .7 and abs(-g_z) < .55;
roll-event counts track excursions above 60 degrees and entries into prone, not
complete 360-degree turns. Side event latch clears below 40 degrees.

Names/order: right hip_pitch, hip_roll, thigh (yaw), calf (knee), ankle_pitch,
ankle_roll; then left in the same order. Pitch/calf/ankle_pitch axes +Y; hip_roll
and ankle_roll +X; thigh +Z. Mirror signs are +,-,-,+,+,-. Body frames bake in
hip pitch -0.25, knee +0.65, ankle pitch -0.40 rad: zero coordinates describe the
nominal bent-knee stance, not straight anatomical legs.

| Joint type | Right range, rad | Left range, rad |
|---|---|---|
| hip_pitch | [-1.25,1.75] | [-1.25,1.75] |
| hip_roll | [-.50,.12] | [-.12,.50] |
| thigh | [-.60,.30] | [-.30,.60] |
| calf | [-.65,1.65] | [-.65,1.65] |
| ankle_pitch | [-.50,1.30] | [-.50,1.30] |
| ankle_roll | [-.30,.80] | [-.80,.30] |

Mass is 6.939736 kg before DR. There are 18 robot collision geoms: body/hip/calf
capsules, thigh spheres and five sole capsules per foot. The nominal zero-angle
stance comes from the existing asset-zoo HOME_KEYFRAME, validated by FK. Sole
contact gives root height **0.345112 m**, not an arbitrary reward height. Sole
centers relative to the upright pelvis are x=0.00379, y=±0.0801 m.

Bounded IK keeps both soles at their nominal XY positions and flat, within physical
limits with a 0.025 rad margin. Candidates with non-foot ground penetration or
self-contact penetration are rejected. Accepted references:

| Pelvis height | hip pitch | knee | ankle pitch | Maximum sole-position error |
|---|---:|---:|---:|---:|
| 0.345112 m | ~0 | ~0 | ~0 | <1e-6 m |
| 0.310601 m | -.26565 | .55266 | -.28702 | <1e-6 m |
| 0.276090 m | -.44723 | .93408 | -.47500 | 0.000594 m |

Other leg coordinates remain near zero. These are an **unordered soft support
set**, never joint targets or a timed motion sequence. The deepest candidate has
about 0.012 rad residual foot pitch (soft IK tolerance). Lower candidates at 70%
and 65% nominal height were not accepted with these fixed sole positions. This
is not proof all deeper crouches are impossible: foot spacing, torso pitch and
contact sequence can change. IK demonstrates static kinematic availability only;
a direct supine path with available motor torque remains unproven.

## Stages, thresholds and weights

Stage numbers below are 1–3 (internal indices 0–2).

1. **Erection**: head/torso height and upright progress. No foot-support requirement.
   Lateral cost has a free region of 30 degrees; prone and rapid off-axis twisting
   incur mild costs. Pitch sit-up itself has neither lateral nor twist cost.
2. **Retraction/support**: soft similarity to the nearest IK support pose, feet XY
   relative to the pelvis, measured foot-ground load, height progress and a mild
   symmetry cost. It does not command final standing angles from lying.
3. **Standing**: near-nominal posture, both feet loaded and aligned, upright body,
   base height, low body velocity and COM projection near the support segment.
   Quality reward ramps with continuous hold; full success requires at least 1 s.

Stage 1→2: up > .5 and base height > .65 * lowest validated support height =
**0.179458 m**, continuously for **0.12 s**. This allows torso erection well before
the feet carry load. Stage 2→3: up > cos(20°), height > .92*nominal = **0.317503 m**,
both foot vertical loads >8% body weight, total >55%, foot normal up >.9, and
maximum joint overshoot <.015 rad, continuously for **0.20 s**.

Per-env stage is monotonic within an episode (no whole-population switching).
Stage-dependent gates blend over **0.4 s**. Loss of balance resets the standing
hold and lowers current geometric gates; the robot can recover while keeping the
latched stage. `regressions` counts losses of support/readiness after stage 3,
not backward stage transitions. No stage-entry reward is paid.

Progress potentials use running maxima, initialized at the first actuated sample.
Each potential contributes at most 2 integrated reward units per episode. Stage-2
progress weight blends from 25% toward 100% after erection. Returning to a previous
pose cannot earn the same progress twice. Holding a crouch produces no new progress;
a small unfinished cost remains, while supported standing pays ongoing reward.

Raw rates in the additive groups (all multiplied by dt=.02 by HoST):

| Term | Rate/weight |
|---|---:|
| Erection progress | 2 * positive high-water increment / dt |
| Support progress | 2 * positive high-water increment / dt * gate |
| Excess sideways inclination | -.15 * smooth bounded severity |
| Prone tendency | -.25 * smooth(g_x,.3,.9) |
| Off-axis angular speed above 2 rad/s | -.03 * bounded severity |
| Physical-limit overshoot | -.30 * clamp(max overshoot/.05,0,2) |
| Stage-2 symmetry | -.10 * support gate * bounded squared asymmetry |
| Not yet valid-standing | -.03 |
| Standing quality and hold | up to +4 |

Original joint-limit regularization is retained. Early non-limit motion penalties
are multiplied by .1, increasing toward 1 late in recovery. This changes their
balance **within regu**; separate advantage normalization means it is not a claim
of a tenfold reduction of their policy-gradient effect. Core task weight stays
2.5; regu .1, style 1 and target 1. Coefficients are initial experimental choices,
not transplanted FTSR coefficients and not evidence of convergence.

Stable-standing validity: tilt <18°, height in [.92,1.10]*nominal, both feet loaded,
foot-up >.94, every joint within .45 rad of nominal, mirrored-leg RMS difference
<.20 rad, instantaneous overshoot <.015 rad, COM outside the sole-center segment's
20mm radius by less than another 25 mm, angular speed <.8 rad/s, body linear speed
<.15 m/s. Validity must hold continuously for 1 s. The COM test is a conservative
geometric support approximation, not a ZMP or torque-feasibility proof.

`Episode_end/success` is the new anatomical 1-second criterion; the baseline
monitor is retained as `Episode_end/legacy_success`. Metrics separately report
sustained upright by any method, anatomical stable
standing, direct stable standing (no >60° side excursion/prone), and stable standing
with whole-episode overshoot <=.015 rad. The latter is a model-bound criterion;
there is still no hardware-calibrated torque-speed envelope or hard velocity cap.

## Validation evidence

Passed:

- CPU and GPU projected-gravity signs for all five canonical orientations.
- 25 sagittal pitch samples: sideways inclination exactly zero.
- Per-env dwell: held upright env reaches stage3 while a lying env stays stage1.
- Fixed-pose transition test: no discrete bonus or reward spike; max raw reward
  change <.1 per step, including hold ramp (<.002 after dt scaling).
- Independent reset clears stage/dwell/blend/hold/high-water and event history.
- Lost contact removes standing reward/hold, reacquisition permits recovery.
- Repeated supine/upright excursions do not regenerate high-water progress.
- A stance with .025 rad joint overshoot fails strict standing.
- Ideal supported pose reward ranking (raw target rate): nominal **4.0**, shallow
  crouch **0.47527**, deeper validated crouch **0.01719**. No ongoing crouch progress.
- Stationary route costs (raw style rate): supine/sagittal sit-up **-.03**, side
  **-.18**, prone **-.28**. Thus a normal pitch sit-up is not treated as a roll.
- Native GPU foot-ground sensor produces nonzero world vertical load at nominal
  contact (example DR sample .637/.703 body weight with 2mm compression). The
  sensor filters against terrain; self-contact cannot count as support.
- Compiled baseline-real and stage plant arrays exactly equal for body mass,
  inertia/frames, joint axes/ranges, collision geometry/masks, armature, actuator
  mapping and torque ranges. Sensors alone add measurements.
- 64 envs, 258 actor features, 12 actions and four reward/critic columns.
- Two direct PPO smoke updates: finite rewards, returns, normalized advantages,
  losses and gradients, with forced timeouts exercising reset code. Loss pairs
  (value,surrogate): (0.29575,0.32369), (70.56661,-0.08472). These test numerical
  plumbing, not learning quality.
- Actual train CLI: two updates, checkpoint save; resume CLI: one additional update
  at iteration2, final checkpoint3. Files live in `/tmp/host_ftsr_stage_smoke*`.
- Evaluation CLI: four envs, one episode on a smoke checkpoint; finite metrics.
- Ruff lint and Python compilation pass for the new package.

No production checkpoint was fine-tuned. Total PPO work was five smoke updates on
64 environments across the validation/CLI tests, not a long run.

## Existing policy trajectory audit

A fresh deterministic mean-action replay reconstructed the existing trajectory
from `host_clpai_realpd_v1/.../model_2950.pt`, with its saved beta, force OFF and DR
OFF on the identical physical model. Original beta-observation jitter remains,
with seeded RNG. See `baseline_2950_trace.csv` (policy-rate samples, actual GPU
foot-ground loads, orientation, stage and reward groups).

Times are from reset; actuation starts after approximately 0.60 s:

| Event | First time |
|---|---:|
| Side inclination >30° | 0.72 s |
| Side inclination >60° | 0.76 s |
| Prone | 0.80 s |
| Legacy upright height/orientation | 1.14 s |

Final base .34299 m and up .98781 look upright, but final joint overshoot is
.02633 rad, and the new standing quality is very low. No strict stable success
was recorded. Peak torque16 Nm, peak joint velocity34.33 rad/s, maximum overshoot
.21600 rad. Torque/velocity/overshoot monitoring samples every5ms (not every2.5ms
physics substep); orientation and stage trace sample20ms. This confirms an early
roll/prone shortcut exists in an available baseline checkpoint. It does not show
that the new reward already fixes it; the hybrid has not undergone long training.

## Commands

From `/home/huy/minipi_getup`, start only when GPU is available:

```bash
.venv/bin/python -m minipi_getup.host_ftsr_stage.train \
  --num-envs 4096 --max-iterations 12000 --seed 1
```

This starts from scratch and creates a new timestamp directory under
`results/host/host_ftsr_stage_v1/`. Checkpoints every50 iterations; Ctrl+C saves
`model_interrupted.pt` (possibly a partial optimizer update). Resume only this task:

```bash
.venv/bin/python -m minipi_getup.host_ftsr_stage.train \
  --num-envs 4096 --max-iterations 12000 --resume PATH_TO_STAGE_CHECKPOINT
```

Validation and force-OFF evaluation:

```bash
.venv/bin/python -m minipi_getup.host_ftsr_stage.validate
.venv/bin/python -m minipi_getup.host_ftsr_stage.validate --gpu
.venv/bin/python -m minipi_getup.host_ftsr_stage.evaluate \
  --checkpoint PATH_TO_STAGE_CHECKPOINT --num-envs 64 --episodes 3 --dr off
```

The generic HoST viewer can replay the unchanged actor with the saved real plant,
but its legacy success display is not the new strict metric; use this evaluator
for the latter. DR-OFF repeated nominal drops are not a broad robustness test.

## Limits and next step

A short controlled scratch run is the next experiment, only when requested.
Compare at fixed beta and force OFF against the recorded baseline: strict/direct
success, roll/prone incidence, joint-stop usage and torque saturation. Do not
interpret legacy upright success or reduced rolling alone as hardware feasibility.
If progress stalls before foot support, audit direct-recovery dynamic feasibility
and foot placement rather than escalating roll penalties. Static IK solutions do
not guarantee a dynamically reachable armless supine-to-standing transition.
