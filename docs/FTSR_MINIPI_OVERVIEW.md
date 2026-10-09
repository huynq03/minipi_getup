# FTSR on Mini-Pi: one-page record of everything done (2026-10-07 → 2026-10-09)

This is the single summary of the `ftsr-only` branch (worktree `/home/huy/minipi_getup_ftsr`).
It covers the work from the first audit to the latest limiter sweep. Each section links to
the detailed record.

Status: **simulation only, nothing hardware-verified, no policy is hardware-ready.**

## 1. Goal and ground rules

- **Goal:** port FTSR, the fall-recovery method of Hou et al. (RA-L 2026, `getup_gym`
  release), to the HighTorque Mini-Pi 12-DOF biped in mjlab 1.6 / MuJoCo Warp. The
  policy must get up from four fallen poses: supine, prone, left side and right side.
- **Read-only sources:**
  - `/home/huy/getup_gym`, the FTSR release.
  - `/home/huy/Hightorque_Pi`, including `mini_pi_fsm`, the deploy runtime.
- **Rules:**
  - Never command the real robot.
  - Commit locally; push only when asked.
  - Every deviation from the release is classified as `SIMULATOR_PORT`, `ROBOT_ADAPTATION`,
    `HARDWARE_CONSTRAINT`, `REFERENCE_BUG_FIX`, `PAPER_COMPLETION`, `PAPER_AMBIGUITY` or
    `UNRESOLVED`.

## 2. Timeline

| Date | Step | Outcome | Detail |
|---|---|---|---|
| 10-07 | Old experiments archived (`archive/ftsr-experiments-20261007`); restart from the pre-FTSR baseline `8a2a990` | clean base | experiment log |
| 10-07 | Audits: FTSR release vs paper, Mini-Pi physical model, deploy contract | several release bugs found; motor data contradictory (blocker) | `FTSR_REFERENCE_AUDIT_V2.md`, `MINIPI_PHYSICAL_MODEL_V2.md`, `MINIPI_DEPLOYMENT_CONTRACT_V2.md` |
| 10-07 | User contract: 0.06 rad/step target slew, 3 rad/s soft envelope; H-conservative motor hypothesis | plant v0–v2 | physical model §4b |
| 10-07 | Port implemented (`src/minipi_getup/ftsr_ref/`), 25/25 tests | `bcd00ee` | `FTSR_REF_IMPLEMENTATION.md` |
| 10-07 | Walk pretrain v1 | accepted at model_900 | experiment log |
| 10-07 | Recovery v1 | 0 % without assistance; stage deadlock; stopped at 1000 | experiment log |
| 10-07/08 | Recovery v2 (r_u height target 0.214) | r_u/r_s limit cycle; v3 monotonic stage prepared, not run; model_2150: 8–22 % without assistance; NaN crash at it 2279 | experiment log, `results/model_2150_*` |
| 10-08 | **pd16_noslew**: plain PD clipped at 16 Nm, no slew, qd envelope 6.28 | new walk + recovery from scratch | `FTSR_PD16_NOSLEW_EXPERIMENT.md` |
| 10-08/09 | Recovery stops at it 2700 (NaN cost from a physics explosion); fix; resume to 8000 | 100 % autonomous recovery from it 4000 | same |
| 10-09 | Joint-velocity audit | peaks 40–50 rad/s are valid motion from large target jumps | `results/ftsr_velocity_audit/` |
| 10-09 | Zero-shot 0.3 rad/step target-rate limiter | model_4000/5000 keep 98 %; −94 % spikes | `results/ftsr_limiter_0p3_evaluation/` |
| 10-09 | Fine-tune with the 0.3 limiter from model_4000, +1000 iterations | **model_4500: 100 % on every pose and seed** | `results/ftsr_limiter_0p3_finetune/` |
| 10-09 | Zero-shot sweep of model_4500 at 0.3 / 0.2 / 0.15 / 0.1 | 100 / 100 / 85 / 31 %; one data record at 0.2 | `results/ftsr_limiter_zeroshot_m4500/` |

## 3. Audits (before any code)

### FTSR release vs paper (`FTSR_REFERENCE_AUDIT_V2.md`)

The executable path is traced line by line. Release robot: 27.7 kg wheeled biped, 120 Nm
legs, 50 Hz policy, 4000 envs (3000 teacher, 1000 student), latent 12, PPO 24 steps,
entropy 0.01, Eq. 4 assist to iteration 3000.

**Bugs and gaps found:**

| Finding | Effect |
|---|---|
| Force-guided optimization (Eq. 5–8) is a no-op | nothing writes `extras["force"]` |
| Teacher-identity shuffle bug | — |
| History stack | holds 3 distinct frames instead of 5 |
| Latent | lags one step |
| Reset pose | only one |
| Termination | none |
| Env step counter | not checkpointed |
| Walking pretrain | missing |

**Paper vs release conflicts:** the critic input and the latent routing disagree. They are
recorded, and the port follows the release.

### Mini-Pi physical model (`MINIPI_PHYSICAL_MODEL_V2.md`)

- **Model:** `cl_pai.xml`, 6.94 kg; vendor URDF ranges and meshes; armature, damping and friction 0.
- **Stance:** all joints at 0 gives a base height of 0.345 m.
- **Joint ranges:**
  - hip pitch [−1.25, 1.75]
  - calf [−0.65, 1.65]
  - ankle pitch [−0.5, 1.3]
  - roll and yaw joints are asymmetric
- **PD gains:** the vendor RL controller and mini_pi_fsm values, kp 60/40/20/60/30/10 and
  kd 2.4/0.8/0.4/2.8/1.6/0.3.
- **Motor blocker:**
  - The installed motors are HTDW-5047-36-NE.
  - The speed numbers in hand (7.85 rad/s no-load) are for another model and unverified.
  - The URDF says 21 Nm / 21 rad/s.
  - Torque caps disagree across sources: 16 / 6 / 35 Nm / none.
  - Two motor hypotheses were set for simulation: H-conservative and H-loose.

### Deploy contract (`MINIPI_DEPLOYMENT_CONTRACT_V2.md`)

- Joint order, signs and offsets of mini_pi_fsm.
- 50 Hz policy, firmware PD, ZOH target.
- Raw clip and per-joint target clip.
- `last_action` = raw-clipped output; history is frame-major with zero-init.
- **No slew limiter exists in mini_pi_fsm.**
- Invariant: q_target is clipped to the physical ranges identically in training and deploy.

## 4. The port (`src/minipi_getup/ftsr_ref/`, `FTSR_REF_IMPLEMENTATION.md`)

### Tasks

| Task | Purpose |
|---|---|
| `Mjlab-FTSR-Ref-MiniPi-Walk` | walking init, `PAPER_COMPLETION` |
| `...-Recovery` | v3: monotonic stages |
| `...-Recovery-Stateless` | v2 semantics; the pd16 runs use it |
| `...-Recovery-Stateless-Limit0p3` | 0.3 rad/step limiter, for training and play |
| `...-Recovery-Stateless-Limit0p1` | 0.1 rad/step limiter, for play and evaluation only |

### Action path

`clip(a, ±50)` → `q_default + scale·a`, with per-joint scales 0.6/0.2/0.2/0.6/0.4/0.25 →
clip to the XML range → optional target-rate limiter → PD held 40 × 0.5 ms.

### Observations

| Group | Dim | Content |
|---|---|---|
| Actor `o_t` | 48 | zero lin-vel slot, gyro, gravity, command, q − q0, qd, last action |
| Student history | 240 | 5 frames |
| Teacher `x_t` | 20 | foot forces, height, root state |
| Critic | 50 + 12 latent | — |

### Episode and assistance

- **Resets:** four fallen poses ± noise, then a **2 s passive window** with kp 0 / kd 1
  (deploy Passive).
- **Assistance (Eq. 4):**
  - F_max = 1.474 m g, scaled from the release.
  - μ and T_max are UNRESOLVED candidates.
  - It ends at iteration 3000.

### Stages and rewards

- h1 / h2 / h3 = 0.19 / 0.276 / 0.345 m, 2/3 rule.
- Release reward formulas with robot replacements.
- Plus a soft qd envelope, −0.01 Σ relu(|qd| − limit)².

### Runner

- Eq. 5–8 actually implemented, with ambiguities resolved and a hand-computed test.
- Release bugs Q1, Q2, Q3 and Q12 fixed. The checkpoint holds networks, both optimizers,
  lr, iteration, env counter and RNG.
- Periodic no-assist evaluation runs in a separate process.

### Validation

`python -m minipi_getup.ftsr_ref.validate` has 34 tests: action contract, PD/cap,
observations, split, stages, assist, Eq. 5–8, resume, ONNX, explosion handling, limiter.
Each change was checked with the relevant subset.

## 5. First plant (v0–v2): motor envelope + 0.06 rad/step slew

- **Plant:**
  - H-conservative torque-speed envelope: ω0 7.85 rad/s, stall 21 Nm, cap 16 Nm.
  - 0.06 rad/step target slew.
  - Physics 0.5 ms, needed because zero armature makes light joints unstable at 2 ms.
- **Walk v1 (model_900):** vx tracking corr 0.999, no falls, torque ≤ 12.8 Nm. It does not
  turn in place (accepted).
- **Recovery v1:** 0 % without assistance; stage deadlock.
  - Cause: the r_u height reward peaks exactly at h1, and Eq. 4 is clamped there.
  - Stopped at 1000.
- **Recovery v2:** one change, the r_u height target set to 0.214 m (release ratio).
  - Reached r_s at it 1251, then cycled r_u ↔ r_s because stateless stages also move the
    assist h_cmd.
  - v3 (monotonic stages) was implemented but not run.
- **model_2150 diagnostic** (`results/model_2150_eval`):
  - Without assistance: 8–22 % recovery.
  - With assistance at tc 0.2: 100 %.
  - The policy depended on the assistance and was aggressive, with joint-limit overshoot
    up to 0.22 rad.
- **v2 crash:** NaN actor parameters at it 2279. There was no guard in the update.

## 6. pd16_noslew plant (current base, `FTSR_PD16_NOSLEW_EXPERIMENT.md`)

### What changed

At the user's request:

- `tau = clip(kp(q* − q) − kd qd, ±16 Nm)`.
- No envelope and no slew.
- qd envelope limit 6.28 rad/s.
- A PPO non-finite guard.

Everything else is the v2 method.

### Runs

- **Walk:** `logs/rsl_rl/minipi_ftsr_ref_walk/2026-10-08_22-59-37_ftsr_walk_pd16_noslew`,
  model_900 accepted (vx corr 1.000, 0 % falls).
- **Recovery:** `2026-10-08_23-51-06_ftsr_recovery_pd16_noslew`, initialized from the walk
  model_900. It stopped at it 2700.
- **Cause of the stop:**
  - One env's physics went non-finite in a substep.
  - mjlab reset that env, but the Eq. 4 cost accumulated for it was NaN.
  - The NaN propagated through `discounted_cost_to_go` into every advantage.
- **Fix** (`9a6796a`; the objective is unchanged on finite data):
  - Sanitize the wrench and cost at the source.
  - Storage guard: invalid samples truncate the trajectory and are left out of
    standardization.
  - Masked PPO loss. When all samples are valid, the result is bitwise identical to the
    code before the fix.
  - Explosion logging and a stop rule.
- **Resume:** `2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700`, 2700 → 8000,
  everything restored. One explosion in 5300 iterations (it 4540, 1 env).

### Periodic evaluations

No assistance, 64 envs per pose:

| it | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 | 7000 | 7500 | 8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recovery | 41 % | 99.6 % | 97.3 % | **100 %** | 100 % | 100 % | 99.2 % | 96.1 % | 94.5 % | **54.3 %** | 99.6 % | 100 % |

- Median get-up time fell from 0.5–1.2 s (it 3000) to about 0.3 s (it 4000–5000).
- The policy std rose from 1.4 to 3.0, and the late policy is unstable (the dip at 7000).
- `best_recovery_model.pt` = model_4000.

## 7. Joint-velocity audit (`results/ftsr_velocity_audit/`)

Data were recorded at every 0.5 ms physics step, for checkpoints 2500–4500.

- **Peaks of 38–47 rad/s are valid motion.** There were 0 invalid episodes out of 1024.
- **Over 99 % of > 20 rad/s events follow a ≥ 0.25 rad target jump.**
  - The PD acts as a velocity servo (qd ≈ kp/kd · error).
  - Free-limb inertia is only 0.0004–0.004 kg m².
- **The policy is bang-bang:** targets sit at the range bounds.
- **The 50 Hz qd penalty is negligible:** about 1.6 reward per get-up, against about 35 per
  second of lying. Training therefore kept speeding up the get-up.
- **Joint-limit pressing while standing** (up to 0.19 rad) is a separate problem.

## 8. Limiter 0.3 rad/step, zero-shot (`results/ftsr_limiter_0p3_evaluation/`)

The limiter: `q* = q*_prev + clip(clip(q_cmd) − q*_prev, ±0.3)`. It starts from the measured
pose at the first actuated step of each episode. Three seeds × 4 poses × 128 episodes,
identical initial states.

| Checkpoint | Recovery, limiter on | Events > 20 rad/s |
|---|---|---|
| 3000 | 93.4 % | −95 % |
| 4000 | 98.2 % | −94 % |
| 5000 | 98.4 % | −94 % |
| 8000 | 50.7 % | — |

- Get-up time increases by 1–1.7 s.
- The maximum joint-limit penetration gets worse (more time spent under load).
- Threshold scan on 4000 / 5000: at 0.25, 88 / 60 %; at 0.2, 71 / 48 %.

## 9. Fine-tune with the 0.3 limiter (`results/ftsr_limiter_0p3_finetune/`)

- **Code** (`48b89cc`):
  - `FtsrActionCfg.target_rate_limit`, default 0.
  - The task `...-Limit0p3`.
  - It shares one function with the evaluation wrapper. Test 33 shows bit-exact agreement,
    including a mid-run reset.
- **Run** `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000`:
  - Starts from model_4000 (sha `408df87e…`) with everything restored.
  - **1000 additional iterations**.
  - Assistance stayed at 0, with 0 NaN, explosions or skipped minibatches.

Three seeds, 1536 episodes per row, limiter 0.3 on:

| | Baseline 4000 + 0.3 | **ft model_4500 (selected)** | ft model_5000 |
|---|---|---|---|
| Recovery / worst pose (min over seeds) | 98.3 / 96.1 % | **100 / 100 %** | 100 / 100 % |
| Strict / foot-only (worst pose and seed) | 95.3 / 95.3 % | **100 / 100 %** | 99.2 / 100 % |
| Median / p95 time | 1.32 / 8.04 s | **0.48 / 1.03 s** | 0.50 / 0.87 s |
| Events > 20 rad/s per episode | 0.85 | **0.19** | 0.29 |
| qd max | 33.5 | 26.6 | 25.2 rad/s |
| Joint-limit penetration max / episodes > 0.05 rad | 0.381 / 86 % | 0.241 / 63 % | **0.155 / 61 %** |

- **model_4500 was selected** for perfect strict and foot-only results and the fewest spikes.
- **model_5000 is the alternative** with the lowest penetration. Its std is rising, and the
  limiter clips more often (13 %).
- **Remaining weaknesses:**
  - The phase p99 stays at about 15 rad/s. This is the allowed target slew: **the limiter
    bounds the PD target, not the joint speed**.
  - Joint-limit pressing remains.

## 10. Zero-shot limiter sweep of model_4500 (`results/ftsr_limiter_zeroshot_m4500/`)

Seed 2150, 512 episodes per level, identical initial states:

| Limit (rad/step) | 0.3 | 0.2 | 0.15 | 0.1 |
|---|---|---|---|---|
| Recovery S / P / L / R (%) | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 99 / 73 / 80 / 89 | 54 / 3 / 31 / 35 |
| Median time (s) | 0.50 | 1.00 | 1.52 | 8.40 |
| Rise attempts of successes, median | 1 | 1 | 2 | 3 |
| Events > 20 per episode | 0.23 | 0.01 | 0.05 | 0.93 |
| Limiter active | 8 % | 14 % | 22 % | 38 % |

### Failures

- **Failure mode:** repeated rise-and-fall, a median of 14–18 attempts. The robot does not
  stall on the ground.
- **Torque saturation** is the same in successes and failures. There is no evidence that
  missing torque is the main cause.
- **Peak base vertical speed** does not separate success from failure, so "lack of
  momentum" is not supported. It is not excluded either.
- **Failures show more limiter clipping and a larger command gap.** This is consistent
  with delayed reactions, but it is a correlation only.

### Record and recommendation

- **Data record (no video):** `records/record_limit0p2_model4500_seed2150_env0_supine*`.
  - It contains initial, control-start and standing joint positions.
  - 50 Hz: q, q_target, q_cmd and base state.
  - 2 kHz: q, qd, tau.
  - It runs from the initial state to 1 s of standing.
- **Recommended first fine-tune level: 0.2.** Its zero-shot result is already 100 % with
  single rises.
- **Not concluded:** whether a curriculum to 0.1 works, and whether the user's target of a
  controlled get-up of about 5 s is reachable with the limiter alone.
  - The 0.3 fine-tune compressed the time back from 1.32 s to 0.48 s, because the reward
    pays for standing early.

## 11. Current state

### Key checkpoints

Local and gitignored; sha256 prefixes:

| Checkpoint | Path | sha256 | Role |
|---|---|---|---|
| Walk init | `logs/rsl_rl/minipi_ftsr_ref_walk/2026-10-08_22-59-37_ftsr_walk_pd16_noslew/model_900.pt` | — | pd16 recovery init |
| model_3000 | `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700/model_3000.pt` | `5ccf02e1` | pd16, 99.6 % |
| model_4000 | same run | `408df87e` | pd16 best (100 %), fine-tune source |
| model_5000 | same run | `7f3f2639` | pd16, 100 % |
| model_8000 | same run | `98abee41` | pd16 final; unstable |
| **model_4500** | `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000/model_4500.pt` | `aea114e1` | **selected, 0.3 limiter** |
| model_5000 (ft) | same ft run | `286e1463` | alternative, 0.3 limiter |

### Play

Run from `/home/huy/minipi_getup_ftsr`, with `WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12`
and `VIRTUAL_ENV` unset. **Use a Limit task for limiter-trained checkpoints.**

```
uv run play Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3 \
  --checkpoint-file logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000/model_4500.pt \
  --num-envs 16 --viewer viser
```

### Tools

| Tool | Use |
|---|---|
| `ftsr_ref/validate.py` | tests |
| `analyze_recovery.py` | periodic evaluation |
| `velocity_audit.py` | substep audit |
| `limiter_eval.py` | `rollout` / `report` / `compare` / `videos_selected` |
| `tools/ftsr_pd16_resume.sh` | resume script |
| `tools/ftsr_limiter_finetune.sh`, `tools/ftsr_limiter_finetune_eval.sh` | fine-tune and its evaluation |

### Open issues, in priority order

1. **Joint-limit pressing.**
   - The targets sit on range bounds 58–67 % of the time, and penetration reaches 0.24 rad
     (model_4500).
   - It needs a target margin or a stiffer limit, as a separate single change.
2. **Slower, controlled get-up** (user goal about 5 s).
   - The next step is a limiter curriculum starting with a 0.2 fine-tune from model_4500.
   - Whether a time or height-reference reward change is needed should be decided from those
     results.
3. **Motor data.**
   - Real no-load speed, rotor inertia and the torque limit are unknown.
   - Simulated peaks of 25–30 rad/s exceed the URDF's 21 rad/s.
   - Hardware feasibility is unverified.
4. **Deploy contract.**
   - Any limiter used in training must be implemented identically in mini_pi_fsm, including
     its initialization from the measured pose. This does not exist yet.
5. **Collision model.** The collision geometry is hand-made (capsules, spheres, 5 foot
   capsules per foot), not the vendor's.
