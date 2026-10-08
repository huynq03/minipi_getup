# HoST Mini-Pi reproduction on MuJoCo/MJLab: `host_reproduction_v1`

**Question.** HoST's Mini-Pi algorithm, rewards, curricula, observations, action
semantics, PPO changes, networks and hyperparameters are kept fixed, and Isaac Gym /
legged_gym is replaced by MuJoCo / MJLab. Does Mini-Pi learn the same stand-up behavior?

**Answer.**

- **It learns to stand up autonomously.** The final policy stands up from supine with the
  pull force off, using a deterministic policy:
  - 100% success without domain randomization (2560/2560 trials);
  - 99.1% sustained-upright success with HoST's domain randomization;
  - about 0.5 s after actuation begins.
- **It does not stand up the way HoST's robot does.** From about iteration 700 onward,
  every successful policy stands on legs **rotated 180° about the vertical axis**:
  - hip pitch ≈ −172° and hip roll ≈ ±172°, at the URDF limits;
  - toes and knees point backwards;
  - the legs whip around at up to ~29 rad/s.

  HoST's release URDF allows this pose, and no HoST reward forbids it. Evidence points to a
  **simulator difference (class B)**: Isaac Gym's joint velocity limit, which MuJoCo lacks.
  It is not proven yet (§7).

Details on how each component maps to MJLab are in `HOST_MJLAB_PORT_AUDIT.md`.

## 1. Files

| Path | What |
|---|---|
| `src/minipi_getup/host/config.py` | `PiCfg` / `PiCfgPPO` and base classes, transcribed from HoST |
| `src/minipi_getup/host/env.py` | `LeggedRobot_Pi` port on MJLab `Scene`/`Simulation`/`Entity`; monitoring; diagnostic knobs `joint_vel_cap`, `limit_solref` (off by default, not used in v1) |
| `src/minipi_getup/host/assets/build_mjcf.py`, `pi_12dof_host.xml`, `pi_12dof_release_v1.urdf` | MJCF generated from HoST's URDF (meshes shared with `asset_zoo`, byte-identical) |
| `src/minipi_getup/host_rl/` | HoST `ActorCritic`, `PPO` (multi-critic + smoothness loss), `RolloutStorage`, `OnPolicyRunner`; additions marked `PORT:` only observe |
| `src/minipi_getup/host/train.py` | `host-train` |
| `src/minipi_getup/host/evaluate.py` | `host-eval` (HoST `eval_ground.py` equivalent, Mini-Pi thresholds) |
| `src/minipi_getup/host/play.py`, `play_viser.py`, `render.py` | mp4 playback and interactive viser playback |
| `src/minipi_getup/host/validate.py` | 12 pre-training checks (`results/host/validation/`) |
| `src/minipi_getup/host/summarize.py` | TensorBoard summary table |
| `HOST_MJLAB_PORT_AUDIT.md` | Source audit and component-by-component equivalence table |
| `results/host/eval/`, `results/host/videos/` | Evaluations, sweeps and videos |
| `pyproject.toml` | `host-train`, `host-eval`, `host-play`, `host-play-viser` scripts |

`results/host/diagnosis/` was produced by a separate session and is not part of this work.

## 2. HoST → MJLab equivalence

**Identical code and math:**
- the action law `q* = q + delay(a·β)`;
- the PD law, the torque clip at ±20 N m, and the actuator DR;
- the β curriculum and the force curriculum;
- the 15 N world-+Z pull force on `base_link`, including its gate and its next-step timing;
- the 43 × 6 observation with its noise, masking, and history that is not cleared at reset;
- all 24 active reward terms, the product task reward, and the 4 groups weighted [2.5, 0.1, 1, 1];
- the terminations;
- the multi-critic PPO: per-group GAE and normalization, smoothness loss, all
  hyperparameters, network sizes and init std 0.8;
- the runner schedule (4096 envs × 50 steps, 12000 iterations, seed 1).

Documented quirks of the release, kept as they are:
- the payload overwrite;
- the zero-observation transition filter;
- stale `time_outs`;
- the per-batch curriculum.

**Different, because of the simulator or its API:**
- **Integration.** Each 5 ms Isaac step is 2 × 2.5 ms MuJoCo steps, with torque and force held
  for the full 5 ms. A single 5 ms step blows up.
- **Contacts.** MuJoCo soft contacts vs PhysX TGS; MuJoCo joint limits are soft.
- **Friction.** One coefficient matched to PhysX's average combine.
- **Not representable in MuJoCo:** restitution and body damping.
- **Joint velocity limit.** None in MuJoCo; possibly 5 rad/s in Isaac.
- **Inertia DR.** Inertia is rescaled with mass, approximating `recomputeInertia`.

## 3. Commands

```sh
cd ~/minipi_getup
export WARP_CACHE_PATH=$PWD/.warp-cache-cu12
# train (as run)
uv run host-train --run-name host_reproduction_v1 --num-envs 4096
# evaluate (HoST eval: deterministic, pull force off, no noise, action scale 0.25, 5 s)
uv run host-eval --checkpoint logs/host/Pi_ground/Oct08_01-49-14_host_reproduction_v1/model_12000.pt \
  --num-envs 512 --episodes 5 --dr off
# one robot, interactive (http://localhost:8080)
uv run host-play-viser --checkpoint logs/host/Pi_ground/Oct08_01-49-14_host_reproduction_v1/model_12000.pt \
  --num-envs 1 --dr off --action-scale 0.25
# one robot, mp4
uv run host-play --checkpoint logs/host/Pi_ground/Oct08_01-49-14_host_reproduction_v1/model_12000.pt \
  --num-envs 1 --pull-force off --dr off --action-scale 0.25
```

## 4. Training run

- **Setup:** 4096 envs (as in the original); 12000 iterations; 9.7 h on one RTX 3090
  (18.4 GB). Training log: `train_host_reproduction_v1.log`.
- **Stability:** no NaN/Inf, no constraint overflow. Zero episodes ended on the
  joint-velocity or base-velocity terminations after the 2.5 ms fix.

| it | task | regu | style | target | base z (m) | success* | time to stand (s) | force (N) | β mean | std |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.05 | −0.69 | −0.15 | 0.01 | 0.14 | 0 | – | 15 | 1.00 | 0.80 |
| 1000 | 0.85 | −2.69 | 0.09 | 0.45 | 0.32 | 0.40 | 0.43 | 0 | 0.86 | 0.39 |
| 3000 | 0.81 | −2.20 | 0.10 | 0.43 | 0.29 | 0.56 | 1.04 | 0 | 0.27 | 0.50 |
| 6000 | 0.87 | −2.49 | 0.16 | 0.57 | 0.32 | 0.64 | 0.83 | 0 | 0.26 | 0.63 |
| 12000 | 0.88 | −2.53 | 0.19 | 0.60 | 0.32 | 0.80 | 0.73 | 0 | 0.25 | 0.69 |

\* Training success means upright at the end of the episode, under DR, observation noise
and a stochastic policy. Group values are per step.

**Progression.** The stages the robot went through:
1. Lying, around iteration 0–200.
2. Torso and head raised; legs whip over and flip, around 200–700.
3. Upright with assistance, around 700.
4. The force curriculum fires. Force goes to 0 N at around 850 and β falls to 0.25 by around 3500.
5. Unassisted stand-up at β = 0.25, from around 2000.
6. Afterwards: faster (1.1 → 0.73 s), more robust (0.56 → 0.80), feet flatter, less drift.

**Mean reward stays around −430.** This is not a failure signal:
- `regu_dof_pos_limits` costs about −2.5 per step. That is the flipped-hip pose sitting
  past the 0.9 × URDF soft limits.
- HoST optimizes per-group normalized advantages, and regu has weight 0.1, so the penalty
  barely moves the policy.

## 5. Results (best/final checkpoint `model_12000`, iteration 12000)

The evaluation follows HoST's `eval_ground.py`: deterministic policy, **pull force off**,
no observation noise, β = 0.25, 5 s episodes, 512 envs × 5 episodes.

| | DR off | DR on (HoST training DR) |
|---|---|---|
| Success, HoST metric (base > 0.317 m once actuated, never < 0.227 m afterwards) | **1.000** | 0.910 |
| Success, sustained (base > 0.317 m and g_z < −0.8 throughout the last 1 s) | **1.000** | **0.991** |
| Stand-up time, from actuation (median / mean) | 0.52 / 0.53 s | 0.62 / 0.66 s |
| Peak torque (max / mean per env) | 9.9 / 9.4 N m | 14.2 / 9.7 N m |
| Joint velocity (mean / peak mean / peak max) | 0.74 / 28.2 / 29.0 rad/s | 0.93 / 23.8 / 37.8 rad/s |
| Energy per 5 s episode | 83 J | 108 J |
| Final base height | 0.351 m | 0.351 m |

Other quantities:
- **Action scale:** β = 0.25 in evaluation. In training at iteration 12000, β had min 0.25,
  mean 0.253 and max 0.9; a few envs are left behind by the per-batch curriculum (§6.3).
- **Policy std:** 0.685.
- **Mini-Pi thresholds:** HoST's G1 thresholds of 0.7 / 0.5 m are scaled by
  0.34 / 0.75 m. Both criteria start counting only once the robot is actuated, because
  Mini-Pi resets at 0.351 m, above the stand threshold.

**Checkpoint sweep** (256 envs; `results/host/eval/sweep_7000_12000.txt`):
- every checkpoint from about 3500 to 12000 reaches 100% with DR off;
- with DR on, they reach 0.96–1.00;
- `model_10500` is equivalent to `model_12000` (DR on, 2560 trials: 0.988 sustained, lower torque);
- `model_10000` reaches 0.984.

**Velocity:** the early policies ran at β = 0.86–1.0 and peaked at 43–48 rad/s with torque at
the 20 N m clip. At β = 0.25 the peak is about 28 rad/s.

## 6. Behavioral findings

### 6.1 The legs are rotated 180° when standing (main discrepancy)

Joint angles of `model_12000` while standing, mean over 64 envs:

| | Hip pitch | Hip roll R / L | Knee | Ankle pitch |
|---|---|---|---|---|
| Angle | −172° | −172° / +172° | −38° | 10° |

- Hip pitch and hip roll sit at the URDF limits of ±171.9°.
- Forward kinematics shows the thigh and foot x-axes point to **−x** while the torso faces +x:
  pitch π followed by roll π equals yaw π.
- During stand-up, hip pitch reaches −184°, which is 12° past the URDF limit. MuJoCo joint
  limits are soft.
- The flip happens in the first 0.2 s, while the robot is still lying, with hip pitch
  going +17° → −172°.
- Every successful checkpoint uses this pose, including iteration 700 at β = 1.0.

Ruled out as causes:
- **Collision.** No geometry blocks the path: HoST's URDF models hip_roll and thigh as 1 mm
  boxes, so a leg can sweep past the torso in Isaac too.
- **Joint-limit softness.** Stiffer limits (solref 0.005) cut the overshoot from 12° to 2°,
  but the pose and the 100% success are unchanged.

### 6.2 Joint velocity is the remaining candidate

The current policy was replayed under PhysX-like caps (`results/host/eval/final/model_12000_posture_and_physx_test.txt`):

| Physics | Stands | Legs flipped | Peak qd |
|---|---|---|---|
| v1 (no cap) | 1.00 | 1.00 | 28.6 rad/s |
| qd cap 10 rad/s | 0.47 | 1.00 | 10 |
| **qd cap 5 rad/s (HoST URDF `velocity`)** | **0.25** | 0.75 | 5 |
| stiff limits | 1.00 | 1.00 | 28.8 |

**Reading:**
- If Isaac Gym enforced the URDF's 5 rad/s through PhysX `maxJointVelocity`, this whip-and-flip
  strategy would mostly fail there, and a policy trained in Isaac would have had to find a
  different one. That is consistent with HoST's real Mini-Pi standing correctly.
- This is a replay of an already-trained policy. Whether training under the cap avoids the
  flip is **not yet tested**.
- Isaac Gym's documentation does not say whether `velocity` is enforced in effort mode.

### 6.3 Other observations

- **Torque is bounded by the control law, not the motors.** At β = 0.25 the
  policy-controlled torque is at most Kp·0.25: 7.5 N m for hip pitch and knee. The policy
  sits at that bound (a ≈ ±1) even while standing still. Peaks of 10–14 N m come from the
  −Kd·qd term and from DR.
- **The per-batch curriculum leaves stragglers.** β drops only when the *batch mean* head
  height exceeds 0.37 m, which is barely below Mini-Pi's standing 0.38 m. About 1% of envs
  stay at β ≈ 0.9. That explains the 60–110 rad/s `Robot/joint_vel_abs_max` in the logs.
- **The settled start state is broad.** Under HoST's DR, the ±1 N m actuation offset during
  the passive window leaves only 36% of envs supine at 0.6 s; 17% are already sitting upright.
- **HoST's real Mini-Pi GIFs** (`HoST/docs/pi_ground.gif` and others) show a fast stand-up.
  They carry no timing, so speed cannot be measured from them.

## 7. Failure analysis and classification

The stand-up task itself did not fail. The discrepancy with HoST's real behavior is the
leg-flip posture.

| Class | Verdict |
|---|---|
| A. API/port mismatch | Two found and fixed before training: a constraint-buffer overflow, and an evaluation off-by-one. No remaining mismatch found: validation passed 12/12, and code, constants and URDF were checked against the source |
| B. Physics-model difference | **Most likely cause of the flip.** Candidate: no joint velocity limit in MuJoCo vs a possible 5 rad/s PhysX limit. Joint-limit softness was ruled out. Also present: the 2.5 ms integration requirement, contact model, and restitution |
| C. Robot-model difference | None vs HoST: the HoST URDF was converted mechanically. The project's real-robot `cl_pai.xml` differs a lot (narrower ranges, 16 N m, other knee/ankle bake), so this policy does not apply to it |
| D. Implementation bug | None found after the fixes in A |
| E. Numerical instability | 5 ms MuJoCo steps diverge; fixed with 2.5 ms substeps. No NaN in the run |
| F. Algorithm does not transfer | Partly. HoST's style terms are binary and its limit penalty is weighted 0.1, so nothing in the reward prevents the flipped pose once the dynamics allow it. HoST's README itself advises narrower joint-deviation ranges for light robots |

**Recommended next experiment** (not run here; it needs about 10 GPU hours): `host_reproduction_v1_velcap`.
- Identical to v1, plus `LeggedRobot_Pi(..., joint_vel_cap=5.0)`, emulating PhysX
  `maxJointVelocity`.
- If the flip disappears, that is the correct MuJoCo stand-in for HoST's physics.
- A second, separate step is HoST on the real-robot plant: `cl_pai.xml` ranges, the motor
  envelope, and the deployment control law. It is needed before any hardware use.

## 8. Videos

All are deterministic with the **pull force off**. The frame strips are the `*_strip.png` files next to them.

| Stage | File |
|---|---|
| early (iteration 1000, β = 0.86, DR off) | `results/host/videos/early_model1000_forceoff_droff_scale0.86_env0.mp4` |
| mid (iteration 3000, β = 0.25, DR off) | `results/host/videos/mid_model3000_forceoff_droff_scale0.25_env0.mp4` |
| **final (iteration 12000, β = 0.25, DR off)** | `results/host/videos/final_model12000_forceoff_droff_scale0.25_env{0,1,2}.mp4` |
| **final, HoST DR on** | `results/host/videos/final_model12000_forceoff_dron_scale0.25_env{0,1,2}.mp4` |
| reset drop / settle (validation) | `results/host/validation/reset_drop_env0.mp4` |
