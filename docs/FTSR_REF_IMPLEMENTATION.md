# FTSR reference port: implementation record (`src/minipi_getup/ftsr_ref/`)

Started 2026-10-07 after the audit at `9791755` was accepted. This file records every
implementation decision with its class (`SIMULATOR_PORT`, `ROBOT_ADAPTATION`,
`HARDWARE_CONSTRAINT`, `REFERENCE_BUG_FIX`, `PAPER_COMPLETION`, `PAPER_AMBIGUITY`,
`UNRESOLVED`). Audits: `FTSR_REFERENCE_AUDIT_V2.md`, `MINIPI_PHYSICAL_MODEL_V2.md`,
`MINIPI_DEPLOYMENT_CONTRACT_V2.md`.

> **2026-10-08, experiment pd16_noslew (`FTSR_PD16_NOSLEW_EXPERIMENT.md`):** the
> motor envelope and the target slew below were removed: the plant is now
> `tau = clip(kp (q* - q) - kd qdot, +-16 Nm)` with no rate limit on q*, and the
> `qd_soft_envelope` limit is 6.28 rad/s. The tables below describe v0-v2.

## Tasks

| Task | Purpose |
|---|---|
| `Mjlab-FTSR-Ref-MiniPi-Walk` | paper Sec. III-A model initialization (r_w only, standing resets, falls terminate, no assist). `PAPER_COMPLETION` (absent from the release) |
| `Mjlab-FTSR-Ref-MiniPi-Recovery` | full FTSR: four fallen poses, passive window, Eq. 4 assist to iteration 3000, stage-wise rewards, Eq. 5–8 |

Same plant, action path, observations and networks in both.

## Plant (nominal: H-conservative)

| Item | Implementation | Class |
|---|---|---|
| Model | `cl_pai.xml` unchanged (6.94 kg, vendor ranges, armature 0) | invariant |
| PD | native `<position>` actuators, kp 60/40/20/60/30/10, kd 2.4/0.8/0.4/2.8/1.6/0.3 (= deploy.yaml) | `HARDWARE_CONSTRAINT` |
| Motor envelope | `tau = clip(tau_PD, L(qd), U(qd))`, U = min(cap, stall(1 − qd/ω0)), L = max(−cap, stall(−1 − qd/ω0)) (= mjlab `dc_motor_clip`); H-conservative ω0 7.85 rad/s, stall 21 Nm, cap 16 Nm; H-loose ω0 21 rad/s (evaluation only) | hypothesis |
| Envelope integration | each physics step the active affine piece (PD, or the curve `±stall − (stall/ω0) qd`) is written to the actuator's gain/bias, so MuJoCo's implicitfast integrates its velocity slope implicitly; forcerange = ±cap | `SIMULATOR_PORT` |
| Physics step | **0.5 ms × 40** = 20 ms policy step | `SIMULATOR_PORT` / `HARDWARE_CONSTRAINT` |
| Armature | none | invariant (no evidence) |
| Passive state | kp 0 / kd 1 on every joint (mini_pi_fsm Passive) | `HARDWARE_CONSTRAINT` |

**Why 0.5 ms (finding of Phase A).** The vendor model has no armature; the ankle-roll
joint inertia is 4.4e-4 kg m² (hip yaw 6.2e-3, ankle pitch 2.8e-3). Under the motor
envelope at 2 ms the light joints run far beyond the no-load speed (floating
single-joint test, slew off: ankle roll 71, ankle pitch 18, hip yaw 13 rad/s for
ω0 = 7.85): the explicit per-step envelope oscillates (stability needs
dt·stall/(ω0 I) < 2; it is 12 for the ankle roll), and the PD accelerates the ankle
roll from rest past ω0 within one 2 ms step. At 1 ms the ankle roll still reaches
29 rad/s. At 0.5 ms calf / hip yaw / ankle pitch saturate at 7.83–7.94 rad/s and the
ankle roll at 9.4 rad/s on raw target jumps (3.7 rad/s with the 0.06 rad slew). The
alternative, a reflected rotor inertia, has no source (mini_pi_fsm `doc/simulation.md`
reaches the same conclusion and runs its simulator at 0.5 ms). Physically the real
36:1 joints certainly have reflected inertia (order 1e-2 kg m² for a 50 mm-class rotor,
which would dominate the ankle-roll link inertia 30×); it stays an **open physical
uncertainty** for a later sensitivity evaluation, not a training parameter. Cost:
~4.2 s per PPO iteration with 4000 envs (RTX 3090).

The previous "500 Hz PD vs 1 kHz firmware" approximation becomes 2 kHz simulated PD
vs a firmware rate ≥ 1 kHz (unknown).

## Action path (identical to the deploy runtime contract)

`a_c = clip(a, ±50)` → `q_cmd = 0 + scale·a_c` → `clip(q_cmd, XML ranges)` →
`q* = clip(q_cmd, q*_prev ± 0.06)` → PD, held 40 physics steps.

| Item | Value | Class |
|---|---|---|
| Absolute target | release | release |
| raw clip | ±50 (release `clip_actions`) | release |
| Per-joint scale | hip pitch 0.6, hip roll 0.2, hip yaw 0.2, knee 0.6, ankle pitch 0.4, ankle roll 0.25: release functional analog (roll 0.2, femur 0.6, tibia 0.6); joints without analog get the release mean scale/range ratio 0.224 × Mini-Pi range | `ROBOT_ADAPTATION` |
| Physical clip | XML ranges = deploy `clip` | invariant |
| Slew | 0.06 rad / 20 ms on the commanded target | `HARDWARE_CONSTRAINT` (user contract) |
| Slew init | measured q clipped, at the first actuated step after reset (end of the passive window / first walk step) | contract |
| last_action | raw-clipped output, before slew; zero at reset and in the passive window | deploy semantics |

## Observations

| Group | Dim | Content |
|---|---|---|
| `actor` (o_t) | 48 | zero lin-vel slot 3, IMU gyro ×0.25, projected gravity, command ×(2, 2, 0.25), q − q0, qd ×0.05, last action (release order/scales); uniform noise gyro 0.0125, gravity 0.05, q 0.04, qd 0.003 |
| `policy` (student) | 240 | o_{t−4..t}, 5 distinct frames (`REFERENCE_BUG_FIX` Q2), frame-major, oldest first, zero-init; o_t = newest frame (single ONNX input) |
| `teacher` (x_t) | 20 | foot forces/(m g) 6, base height ×0.4/0.46, root pos − env origin 3, quat (w ≥ 0) 4, lin vel world 3, ang vel world 3 |
| `critic` | 50 (+12 latent) | noise-free o_t with true lin vel ×2, one height sample `clip(h − 0.23, ±1)·5` (flat ground: the release's 187 samples are identical, `SIMULATOR_PORT`), base height |

All groups zero during the passive window (release Q14 from iteration 0, matching the
deploy history_init zero). No gait clock (to be added only if walking fails without).

## Resets and passive window

Four fallen poses (`PAPER_COMPLETION` Q7) + U(±0.3) roll/pitch, uniform yaw, joints
U(±0.3) additive (`ROBOT_ADAPTATION`), dropped from 0.16 m. **Passive window 100 steps
= 2.0 s** (release 0.6 s): under deploy Passive (kd 1) the side poses still roll at
0.6 s (p95 0.3 m/s); at 2.0 s all poses are at rest (p95 |v| ≤ 0.03 m/s, |ω| ≤ 0.4
rad/s). `ROBOT_ADAPTATION`. Settled heights 0.069–0.083 m.

## Assistance (Eq. 4, `PAPER_COMPLETION`)

| Parameter | Value | Derivation / class |
|---|---|---|
| F_max | 100.3 N = 1.474 m g | release 400 N / (27.669 kg g) × Mini-Pi m g (`ROBOT_ADAPTATION`); ablation 1.0 m g planned |
| μ | 19.2 /m | UNRESOLVED candidate: release height factor at its reset state (0.65/0.75 = 0.867) reproduced at Mini-Pi lying (0.085 m) under h1 = 0.19 |
| T_max | 0.215 N m/rad | UNRESOLVED candidate: release torque at its reset state, 10·(1 − √½)·0.867 = 2.54 N m = 0.0125 m_ref g L_ref, scaled by m g L at tilt π/2 |
| t_tag | 3000 × 24 env steps | release; exactly zero after (asserted, test 20) |
| Applied | base_link, every physics step, zero in the passive window | release (per physics step) |

Observation (test 19): with F_max > m g the assistance alone lifts a lying robot that
holds its entry pose to 0.32 m within 1 s once the population's stage advances (the
equilibrium sits 0.06 m below each stage's h_cmd). The release's linear form also
floats its robot (1.28 m g at the reset height). Kept as derived; the 1.0 m g ablation
tests it.

## Stages

Paper text (S_1 = {h > h1_cmd}, S_2 = {h > h2_cmd}; thresholds = the earlier stages'
targets), release frequency (every env step, stateless, can regress). Candidates:
h1 = 0.19 m (release r_s switch 0.40 m ↦ 0.184 m, and Mini-Pi's lowest balanced
crouch 0.19 m), h2 = 0.276 m (0.8 L: the release's 0.6 m ladder rung and velocity
gate), h3 = 0.345 m (stance). Stateless ⇒ nothing to checkpoint.

## Rewards

Release g2/g3/g4 formulas and weights, robot replacements only; table in
`config/rewards.py`. Plus the user's soft joint-speed envelope `Σ relu(|qd| − 3)²`,
weight −0.01 in all stages (`HARDWARE_CONSTRAINT`). No termination in recovery
(release); walk: fall termination (h < 0.15 m or tilt > 1 rad).

## Eq. 5–8 (resolved ambiguities, hand-computed test 21)

`Ā = std(A) − Σ_i β_i (J_t,i + std(A_C,i)/(1 − γ))`, β = 0.001:

1. A_C: GAE with zero baseline (λ-discounted cost sum); no cost critic (none in paper
   or release; GAE does not need a learned baseline).
2. J_t: discounted cost-to-go from t (truncated at episode/rollout end).
3. Standardization: A and each A_C over the joint batch; J not standardized.
4. Costs: |F|/F_max and |T|/(T_max π), per-step mean over physics steps (hypothesis):
   with β = 0.001 the J term ≤ 0.1 and the A_C term 0.1 per standard deviation; with
   the paper's degrading β = 0.02 both reach 2, i.e. dominate the unit-variance A.
5. One joint batch for both groups (release).

## Runner

Release TSPPO/EncoderMSE semantics; bug fixes Q1 (per-sample teacher mask), Q3
(same-step latents), Q12 (checkpoint: networks, both optimizers, adaptive lr,
iteration, env step counter, RNG). Critic = `[stored E_t(x_t), s_t]` for every row
(release). One optimizer and grad clip 1.0 over actor, std, critic, teacher encoder.
Periodic no-assist evaluation spawned every 500 iterations (recovery).

## Validation (Phase A)

`python -m minipi_getup.ftsr_ref.validate`: 25/25 pass (243 s).
