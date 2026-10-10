# HoST on Mini-Pi: one-page record of everything done (2026-10-08)

This is the single summary of the HoST work. It covers the work from the port of HoST's
`pi_ground` task to MJLab through the last HoST + FTSR-stage pilot.

- **Branches.**
  - `host-only` (this branch) holds the port, the reproduction and the narrow-range run.
  - `ftsr-host` holds the later runs on the real-robot model (`cl_pai.xml`): §7–§9.
- **Detailed records:**
  - `HOST_MJLAB_PORT_AUDIT.md`;
  - `HOST_REPRODUCTION_REPORT.md`;
  - `results/host/host_ftsr_stage_v1/REPORT.md` on `ftsr-host`.

Status: **simulation only, nothing hardware-verified, no HoST policy is hardware-ready.**

## 1. Goal and ground rules

- **Question.** HoST (Huang et al., "Learning Humanoid Standing-up Control across Diverse
  Postures") releases a Mini-Pi task, `pi_ground`. Keep its algorithm fixed:
  - rewards and curricula;
  - observations and action semantics;
  - its PPO changes, networks and hyperparameters.

  Replace only Isaac Gym / legged_gym with MuJoCo / MJLab. Does Mini-Pi learn the same
  stand-up?
- **Read-only reference:** `/home/huy/HoST`, untouched.
- **Ground truth:** the release source code, not the paper or README.
- **Rules:**
  - Never command the real robot.
  - Never edit the reference repositories or the original assets.
  - Each experiment gets its own asset or config copy.

## 2. Timeline

| Step | Run / commit | Outcome |
|---|---|---|
| Port HoST `pi_ground` to MJLab with HoST's RSL-RL fork | `d2788a2` | 12/12 validation checks pass |
| Evaluation, video tools, port audit | `a5bcc9c`, `beadf92` | `HOST_MJLAB_PORT_AUDIT.md` |
| **Reproduction**: 12000 iterations on HoST's own URDF | `host_reproduction_v1`, `f1827a0` | 100 % stand-up (DR off), 99.1 % (DR on), but with the legs rotated 180° |
| HoST URDF with the real robot's joint ranges | `host_supine_narrow_v1`, `f9576ba` | rises to standing height but tilted (~19°), pressing the joint stops; strict 0 % |
| HoST on `cl_pai.xml`, HoST gains, 20 N m | `host_clpai_narrow_v1` (`ftsr-host`) | never stands in 480 iterations; stopped |
| HoST on `cl_pai.xml`, real PD and 16 N m | `host_clpai_realpd_v1` (`ftsr-host`) | 60 % legacy success at 2950, β stuck near 0.94, roll/prone shortcut |
| HoST + FTSR-style per-env stages | `host_ftsr_stage_v1` (`ftsr-host`) | upright 70–85 %, **strict stable 0 %** to 4200 |
| Stage v2: reachable Stage 3 entry and a joint-margin reward | `host_ftsr_stage_v2` pilot (`ftsr-host`, `f9fd654`) | strict stable 0 % at 1220; stopped |

After this the work moved to the FTSR port (`ftsr-only` branch, `docs/FTSR_MINIPI_OVERVIEW.md`).

## 3. The port (`src/minipi_getup/host/`, `src/minipi_getup/host_rl/`)

- **Environment.**
  - The release's `LeggedRobot_Pi` is transcribed method by method into `host/env.py`.
    It keeps the buffer names, the call order and the formulas.
  - It runs on MJLab's `Scene`, `Simulation` and `Entity` primitives, not on the
    manager-based env. That env's fixed step order cannot express HoST's semantics:
    - curriculum before reset;
    - rewards scaled by dt;
    - a single reward vector;
    - observation history cleared at reset;
    - no hook between physics substeps.
- **Config.** `PiCfg` and `PiCfgPPO` are copied into `host/config.py` with the same class
  inheritance.
- **RL.** HoST's `ActorCritic`, `PPO` (multi-critic plus smoothness loss) and
  `RolloutStorage` are copied verbatim. The runner gets logging- and diagnostics-only
  additions, marked `PORT:`.
- **Asset.** `host/assets/pi_12dof_host.xml` is generated mechanically from HoST's
  `pi_12dof_release_v1.urdf` (6.92 kg, 20 N m, 5 rad/s). Its meshes are byte-identical.
- **Commands.** `host-train`, `host-eval`, `host-play` and `host-play-viser` in
  `pyproject.toml`, plus `host/validate.py` and `host/summarize.py`.

### Identical to the release

- the action law `q* = q + delay(a·β)`;
- the PD law, the ±20 N m clip and the actuator DR;
- the β (action-scale) curriculum and the force curriculum;
- the 15 N world-+Z pull force on `base_link`, with its gate and next-step timing;
- the 43 × 6 observation with its noise, masking, and history that is not cleared at reset;
- all 24 reward terms, the product task reward, and the 4 groups weighted [2.5, 0.1, 1, 1];
- the terminations;
- multi-critic PPO with per-group GAE and normalization, the smoothness loss, and every
  hyperparameter;
- the schedule: 4096 envs × 50 steps, 12000 iterations, seed 1.

Release quirks are kept on purpose:
- the payload overwrite;
- the zero-observation transition filter;
- stale `time_outs`;
- the per-batch curriculum.

### Different, because of the simulator

| Item | Difference |
|---|---|
| Integration | Each 5 ms Isaac step is 2 × 2.5 ms MuJoCo steps, with torque and force held for 5 ms. A single 5 ms step diverges (qd up to 5e4 rad/s). |
| Contacts | MuJoCo soft contacts vs PhysX TGS; MuJoCo joint limits are soft. |
| Friction | One coefficient matched to PhysX's average combine. |
| Not representable | Restitution and body damping. |
| Joint velocity limit | MuJoCo has none. Isaac Gym may enforce the URDF's 5 rad/s; the docs do not say. |
| Inertia DR | Inertia is rescaled with mass, approximating `recomputeInertia`. |

### Validation

12/12 checks pass (`python -m minipi_getup.host.validate`):
- timing and the 30-step unactuated window;
- first actuation at step 32;
- action semantics (0.1 rad at β = 1, 0.025 rad at β = 0.25);
- the delay buffer;
- observation layout and the noise vector;
- the pull force (COM acceleration matches F/M to 0.03 %);
- finite rewards in lying and standing states;
- multi-critic shapes and normalization;
- the supine reset.

Two port bugs were found and fixed before training:
- a constraint-buffer overflow, fixed with `njmax 1200` and `nconmax 300`;
- an evaluation off-by-one.

## 4. Reproduction: `host_reproduction_v1` (HoST URDF, 12000 iterations)

- 4096 envs, 9.7 h on one RTX 3090.
- No NaN, no constraint overflow, zero velocity terminations after the 2.5 ms fix.
- **Progression:**
  1. Lying.
  2. Torso raised, legs whipping over, around iterations 200–700.
  3. Upright with assistance at about 700.
  4. The pull force reaches 0 at about 850; β reaches 0.25 at about 3500.
  5. Unassisted stand-up from about 2000.
  6. Then faster and more robust.

**Evaluation** follows HoST's `eval_ground.py`: deterministic, **pull force off**, no noise,
β = 0.25, 5 s episodes, 512 envs × 5 episodes. Thresholds are scaled to Mini-Pi: stand
0.317 m, fall 0.227 m.

| `model_12000` | DR off | HoST DR on |
|---|---|---|
| HoST success | **100 %** | 91.0 % |
| Sustained (> 0.317 m and g_z < −0.8 over the last 1 s) | **100 %** | **99.1 %** |
| Stand-up time, median | 0.52 s | 0.62 s |
| Peak torque max | 9.9 N m | 14.2 N m |
| Peak joint velocity max | 29.0 rad/s | 37.8 rad/s |

Checkpoints from about 3500 to 12000 all reach 100 % with DR off. `model_10500` is
equivalent to 12000.

### Main discrepancy: the legs stand rotated 180°

- From about iteration 700 on, every successful policy stands with:
  - hip pitch ≈ −172° and hip roll ≈ ±172°, at the URDF limits;
  - toes and knees pointing backwards.
- The flip happens in the first 0.2 s while still lying (hip pitch +17° → −172°).
- HoST's URDF allows the pose, and no HoST reward forbids it: the style terms are binary,
  and the limit penalty has weight 0.1.
- **Ruled out:**
  - collision: HoST models hip_roll and thigh as 1 mm boxes;
  - joint-limit softness: stiffer limits cut the overshoot 12° → 2°, but the pose is
    unchanged.
- **Remaining candidate: the joint velocity limit.** Replaying `model_12000`:

| Physics | Stands | Legs flipped | Peak qd |
|---|---|---|---|
| no cap (v1) | 100 % | 100 % | 28.6 rad/s |
| qd cap 10 rad/s | 47 % | 100 % | 10 |
| qd cap 5 rad/s (URDF `velocity`) | 25 % | 75 % | 5 |
| stiff limits | 100 % | 100 % | 28.8 |

**Conclusion.**
- The algorithm transfers: Mini-Pi learns to stand up in MJLab.
- The posture does not match HoST's robot. The most likely cause is a simulator
  difference (class B, a PhysX joint velocity cap), but it is **not proven**.
- A training run with the cap (`joint_vel_cap=5.0`) was proposed and not run.

## 5. Narrow ranges on the HoST URDF: `host_supine_narrow_v1`

- **Change.**
  - The 12 joint ranges of HoST's URDF are replaced by `cl_pai.xml`'s, for example hip
    pitch [−1.25, 1.75] and hip roll −0.5..0.12 / −0.12..0.5.
  - `jnt_range` is the only compiled difference; this is checked.
  - The asset is `results/host/host_supine_narrow_v1/assets/pi_12dof_host_narrow.xml`.
    The original is unchanged (sha recorded).
  - Everything else is v1: HoST gains, 20 N m, rewards, curricula.
- **Training.**
  - From scratch to 460 iterations within a 25 min budget (`training.csv`).
  - Resumed to 4315 (`resume.py`, `resume_20261008_121442_690777/`).
  - The original checkpoint had no per-env curriculum state, so the curriculum was
    re-initialized from the logged means.
- **TensorBoard at 4315:**

  | Episode_end/success | base height | force | β |
  |---|---|---|---|
  | 0.38 | 0.32 m | 0 N | 0.34 |

**Convergence check** (`convergence_trials.csv`, `convergence.png`): 64 trials per
checkpoint, β = 0.25, pull force off.

| it | strict held 1 s | geometric final 1 s | max base (m) | final tilt | time at joint stop |
|---|---|---|---|---|---|
| 3150 | 0 % | 6 % | 0.338 | 55° | 86 % |
| 3400 | 0 % | 69 % | 0.356 | 26° | 95 % |
| 3650 | 0 % | 98 % | 0.359 | 20° | 99 % |
| 3900 | 0 % | 100 % | 0.360 | 19° | 99 % |
| 4150 | 0 % | 100 % | 0.359 | 18° | 99 % |

Peak qd is about 23–24 rad/s.

**Reading.**
- Without the 180° flip the policy still reaches standing height.
- It ends tilted about 19°, with joints pressed against a stop almost all the time, and
  never meets the strict criterion.
- The strict and geometric criteria were defined in an evaluator that is not in the
  repo. Only their outcome columns are recorded.
- This is still HoST's plant (gains, 20 N m), so it is not the real robot.

## 6. Moving to the real-robot model (`cl_pai.xml`) — branch `ftsr-host`

- **Assets.** `host/clpai_asset.py` loads the asset-zoo `cl_pai.xml` in memory. It adds
  HoST's 9 massless, geometry-free measurement frames: the head keyframe and 4 auxiliary
  frames per foot. No XML is written.
- **Torque limits.** `host/env.py` gains `TORQUE_LIMITS`, per-joint-type limits. The
  release default is unchanged: `None` keeps 20 N m everywhere.
- **Launcher.** `host/train_clpai.py`. `--plant real` uses:
  - the vendor gains, kp 60/40/20/60/30/10 and kd 2.4/0.8/0.4/2.8/1.6/0.3;
  - `pai_model.yaml` torque limits: 16 N m, and 6 N m for the ankle roll.

  `--plant host` keeps HoST's gains and 20 N m.

## 7. `host_clpai_narrow_v1` and `host_clpai_realpd_v1`

| Run | Plant | Iterations | Result |
|---|---|---|---|
| `host_clpai_narrow_v1` | cl_pai, HoST gains, 20 N m | 0–480 | success 0. The force curriculum never fires (15 N throughout, β = 1). Base height falls back to about 0.11 m. Stopped. |
| `host_clpai_realpd_v1` | cl_pai, real PD, 16 / 6 N m | 0–2985 | Legacy success 0.08 (750) → 0.44 (1500) → 0.61 (2985). The force reaches 0 at about 1500. **β stays at 0.94–0.97**, because the curriculum height threshold is rarely met. Peak qd is about 50 rad/s. |

A deterministic replay of `realpd_v1/model_2950` (force off, DR off) shows:

| Event | First time |
|---|---|
| Side inclination > 30° | 0.72 s |
| Side inclination > 60° | 0.76 s |
| Prone | 0.80 s |
| Upright | 1.14 s |

- It rolls over onto its front and pushes up from prone. It is a roll/prone shortcut, not
  a direct supine get-up.
- Final base is 0.343 m, but the final joint overshoot is 0.026 rad and the maximum
  overshoot is 0.216 rad.
- Peak torque is 16 N m and peak qd 34 rad/s.

The trace is in `results/host/host_ftsr_stage_v1/baseline_2950_trace.csv`.

## 8. HoST + FTSR-style stages: `host_ftsr_stage_v1`

Code: `src/minipi_getup/host_ftsr_stage/`. Full record: `results/host/host_ftsr_stage_v1/REPORT.md`.

- **Unchanged from the realpd baseline:**
  - plant, PD, the 16 N m limits, the action law and the β/pull curricula;
  - the 258-dim actor observation;
  - the four critics and their weights [2.5, 0.1, 1, 1];
  - DR and the initial state.

  **No privileged features were added** to the actor or the critics.
- **Added: per-env, monotonic stages** with dwell and 0.4 s blending (not population
  switching).
  1. **Erection:** high-water progress on height and uprightness. Mild costs for
     sideways tilt over 30°, prone and twisting. A sagittal sit-up costs nothing.
  2. **Support:** soft similarity to IK-validated support poses at 0.345, 0.311 and
     0.276 m, foot load, and symmetry.
  3. **Standing:** near-nominal posture, both feet loaded, upright, COM over the support,
     and a 1 s hold.
- **Strict success:** stable standing for 1 s with:
  - tilt < 18°;
  - joints within 0.45 rad of nominal;
  - left/right symmetry RMS < 0.2 rad;
  - overshoot < 0.015 rad.

  The old HoST monitor is kept as `legacy_success`.
- **Validation:**
  - frame and projected-gravity signs on CPU and GPU;
  - dwell, reset and high-water behavior;
  - no reward spikes;
  - plant parity with the baseline;
  - PPO smoke updates;
  - CLI train/resume/eval.
- **Training to 4246:**

  | Metric | Value |
  |---|---|
  | legacy success | 0.63–0.75 |
  | upright by any method | 0.72–0.85 |
  | **strict stable / direct / physical success** | **0 throughout** |
  | prone fraction | 3–9 % |
  | max side tilt | ~54° |
  | β | 1.0 → 0.52 |

**Evaluation of v1 checkpoints** (deterministic, force off, 256 samples,
`results/host/host_ftsr_stage_v2/eval/`):

| | 1000 DR off | 1000 DR on | 2250 DR off | 2250 DR on | 2250 at β 0.25 |
|---|---|---|---|---|---|
| β | 0.99 | 0.99 | 0.77 | 0.77 | 0.25 |
| upright | 100 % | 82 % | 100 % | 88 % | 0 % |
| strict stable | 0 | 0 | 0 | 0 | 0 |
| time to stand, median | 0.48 s | 0.48 s | 0.58 s | 0.54 s | – |
| max overshoot | 0.21 rad | 0.32 | 0.19 | 0.26 | 0.09 |
| final overshoot, median | 0.023 rad | 0.024 | 0.011 | 0.011 | 0.020 |
| final left/right symmetry RMS | 0.63 rad | 0.63 | 0.63 | 0.63 | 0.93 |
| near a joint limit | 99.6 % | 99.99 % | 99.6 % | 99.99 % | 98.9 % |
| peak qd p95 | 32 rad/s | 40 | 26 | 33 | 11 |

**Reading.**
- The policy stands up quickly (about 0.5 s) at high β.
- The stance is asymmetric (RMS 0.63 rad) and sits on joint limits.
- It never passes the strict check, so the overshoot and symmetry gates block Stage 3.
- At the target β = 0.25 it does not stand at all.

## 9. `host_ftsr_stage_v2` pilot (`src/minipi_getup/host_ftsr_stage_v2/`, `f9fd654`)

- **Changes vs v1:**
  - Stage 3 entry tolerates overshoot up to 0.03 rad. Strict success keeps 0.015.
  - A standing-gated `target_joint_margin` reward (weight 1.0) for staying ≥ 0.05 rad
    inside each limit.
- **Pilot,** resumed at about 1000 and run to 1220:

  | Metric | Value |
  |---|---|
  | legacy success | 0.75–0.80 |
  | upright by any method | swings 0.0–0.9 |
  | **strict stable** | **0** |
  | force | stuck at 8.2 N |
  | β | 0.99 |

  Stopped there. No conclusion beyond "no strict success in the pilot".
- Checkpoints, logs and videos stay local (gitignored). Configs, TensorBoard events and
  eval JSONs are committed.

## 10. What was learned

1. **The HoST algorithm transfers to MJLab** on HoST's own robot model. It reaches
   100 % / 99.1 % unassisted stand-up in about 0.5 s.
2. **The behavior HoST learns depends on what the model allows.**
   - With HoST's wide URDF ranges it exploits a 180° leg flip.
   - With the real ranges it stands tilted against the joint stops.
   - On the real plant (`cl_pai.xml`, real PD, 16 N m) it rolls to prone and pushes up,
     at β near 1.
3. **Nothing reached strict, symmetric, in-range standing on the real model.** Stage
   shaping (v1, v2) did not change that within the iterations run.
4. **The work then moved to the FTSR port** (`ftsr-only`), on the same real plant. That
   port has absolute PD targets, recovery stages and force guidance. Whether HoST's
   relative action law (`q* = q + β·a`) or its curricula caused the failures above was
   not tested.

## 11. Checkpoints (gitignored, local in `/home/huy/minipi_getup`)

| Run | Path |
|---|---|
| reproduction v1 | `logs/host/Pi_ground/Oct08_01-49-14_host_reproduction_v1/model_12000.pt` (also 10500) |
| narrow v1 | `results/host/host_supine_narrow_v1/logs/model_460_final.pt`; resume `resume_20261008_121442_690777/model_4300.pt` |
| clpai narrow v1 | `results/host/host_clpai_narrow_v1/20261008_153100_024556/model_450.pt` |
| clpai realpd v1 | `results/host/host_clpai_realpd_v1/20261008_155406_448018/model_2950.pt` |
| stage v1 | `results/host/host_ftsr_stage_v1/20261008_183136_920453/model_4200.pt` (evaluated: 1000, 2250) |
| stage v2 pilot | `results/host/host_ftsr_stage_v2/20261008_193713_323041/model_1200.pt` |

Play the reproduction:

```sh
cd ~/minipi_getup && export WARP_CACHE_PATH=$PWD/.warp-cache-cu12
uv run host-play-viser --checkpoint logs/host/Pi_ground/Oct08_01-49-14_host_reproduction_v1/model_12000.pt \
  --num-envs 1 --dr off --action-scale 0.25
```

## 12. Open issues (not run)

- **HoST v1 with `joint_vel_cap=5.0`.** Test whether the 180° flip is the missing PhysX
  velocity cap. It needs about 10 GPU hours.
- **Joint-limit pressing and asymmetry on the real model.** These are unsolved; the same
  weakness appears in the FTSR port.
- **β curriculum on the real plant.** The HoST threshold (head 0.37 m) is barely
  reachable, so β stays near 1.
- **Hardware.** No HoST policy uses the deploy control law of `mini_pi_fsm`. None is
  hardware-ready.
