# FTSR reproduction audit (Mini-Pi, MjLab)

Audit written on 2026-10-07, before any code change for the reproduction.

Sources:

- **Paper**: Hou et al., "Robust Fall Recovery for Armless Bipedal-Wheeled Robots via
  Force-Guided Learning" (RA-L, accepted May 2026). It is the primary source for the
  method.
- **Reference code**: `/home/huy/getup_gym` (Isaac Gym, read-only). Paths below are
  relative to `/home/huy/getup_gym/getup_gym/`.
- **Target**: `/home/huy/minipi_getup` (mjlab 1.6.0, MuJoCo Warp, rsl-rl-lib 5.4.2).

## 1. Repository state at audit time

| Repo | Branch | HEAD | Working tree |
|---|---|---|---|
| `/home/huy/minipi_getup` | `main` (tracks `origin/main`) | `8a2a990 test` | clean |
| `/home/huy/getup_gym` | `master` (tracks `origin/master`) | `238d296 improve humanoid training` | clean, never modified |

The machine is an RTX 3090 (24 GiB, sm_86) on Ubuntu, with 62 GiB RAM, 20 CPUs and
347 GiB free disk. No training process was running. The NixOS / GTX 1080 Ti notes in
`CLAUDE.md` don't describe this machine. Env-building scripts need
`WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12` (see the user memory note).

### Existing Mini-Pi tasks (kept unchanged)

`src/minipi_getup/getup/config/minipi/__init__.py` registers:

- `Mjlab-Getup-Flat-MiniPi`: the baseline get-up. Supine plus crouch resets, relative
  joint-position actions at 0.25 rad, 16 Nm effort limit.
- `Mjlab-Getup-Flat-MiniPi-ForceGuided`, `...-ForceGuidedReleased`,
  `...-ForceGuidedStand` and `...-ForceGuidedSafe`: earlier FTSR-*inspired* attempts.
  - `getup/mdp/force_guidance.py` holds an Eq. 4 assist event, force/torque as plain
    negative rewards, and a `height_stage` curriculum.
  - They use the stock rsl-rl PPO. There is no teacher-student split, no
    constrained-advantage term, no walking stage, no pretraining, and only the supine
    reset.
  - The runs in `logs/rsl_rl/minipi_getup/*ftsr*`, `*safe_v*` ended with a leaping
    get-up: 16 Nm saturation, 22 rad/s joint speed, 2.3 m/s vertical base speed.
    That is the "jerk upright" behaviour this reproduction must avoid.

None of these are modified. The reproduction is a new task family.

## 2. Mini-Pi physical quantities (measured, not assumed)

Measured with `mujoco` on the compiled `cl_pai.xml` used by the task (12 actuated
joints, 14 bodies):

| Quantity | Value | How |
|---|---|---|
| Total mass | 6.94 kg (weight 68.1 N) | `sum(body_mass)` |
| Torso (`base_link`) mass | 2.16 kg | XML |
| Standing base height (all joints 0) | 0.345 m | kinematic: base origin above lowest sole point. Matches `_TORSO_HEIGHT = 0.344` |
| Standing CoM height | 0.223 m | `subtree_com` |
| Composite inertia about CoM (standing) | (0.180, 0.153, 0.041) kg·m² | parallel-axis sum |
| Lowest statically balanced crouch | base ≈ 0.19 m | `env_cfgs._SQUAT_POSES` (kinematic search) |
| Supine resting base height | 0.085 m | `env_cfgs.SUPINE_ROOT_POS` |
| Prone resting base height | ≈ 0.084 m | settle test |
| PD gains (task, 0.7× vendor kp) | kp 42/28/14/42/21/7, kd 2.4/0.8/0.4/2.8/1.6/0.3 | hip pitch / hip roll / hip yaw / knee / ankle pitch / ankle roll |
| Effort limit in asset | 16 Nm (all joints) | `minipi_constants.MINIPI_EFFORT_LIMIT` |

Side-lying poses with straight legs aren't statically stable: in the settle test the
robot rolls toward its back or front. That is physical, and the reset distribution
lets it happen.

## 3. Paper items vs. reference code

| Item | Paper | getup_gym (released code) | Reproduction decision |
|---|---|---|---|
| **Algorithm 1** | Eq. 4 forces → costs; GAE; mixed advantage Eq. 8; stage via `p`; PPO + value update; student MSE | `runners/on_policy_runner.py` `learn()`: TS-PPO update then `EncoderMSE.update()` every iteration | Same order: rollout → GAE (reward and 2 costs) → Eq. 8 → PPO → student MSE |
| **Eq. 4 force/torque** | `[F;T] = (1-exp(-μ(h_cmd_i - h))) · sat01(1 - t/t_tag) · [F_max n; T_max log(R_tag R⁻¹)]`, `h` = CoM height | `envs/legged_robot.py::_base_force_pull_up`: **linear** `(h_target-h)/h_target`, always vs. the final height (not stage `h_cmd_i`), clamp `[0,F_max]`, `F_max` = 400 N (wheeled) / 350 N (G1), torque only about x with an ad-hoc `|q_y+1|` scale | Paper form (exponential, stage-dependent `h_cmd_i`, so₃ log map), clamped at 0 above the target (the paper formula would turn negative and pull down). `h` = torso height, as in the code and the stage sets (documented deviation from "CoM") |
| **Time coefficient** | `t_coeff = 1 - t/t_tag`, `t` in env steps | `(pull_up_end_epochs - (step-1)/24) / pull_up_end_epochs` | Same, `t_tag = 3000 × 24` env steps |
| **Assist end** | ~3000 iterations; constraint bound `d_i = 0` at `t_tag` | `pull_up_end_epochs = 3000` (wheeled), 5000 (humanoid) | 3000 iterations. Exactly zero afterwards, verified in logs |
| **Eq. 5–8 constrained objective** | Penalty method: `Ā = A − Σ β_i (J_Ci + A_Ci/(1−γ))`, GAE for both, advantages standardized | `storage/rollout_storage.py::compute_returns` adds `F_t − γF_{t+1}` to the TD error (potential shaping). The env **never writes** `extras["force"]`, so the term is always zero: **inactive** | Cost critic (2 outputs: force, torque), cost-GAE, standardized cost advantages, `Ā = Â_r − Σ β/(1−γ) Â_Ci`. The `J_Ci` term is a batch constant and is dropped from the gradient (logged instead). The term turns off when the rollout's costs are all zero (`d_i = 0` after `t_tag`) |
| **Penalty factor β** | 0.001 baseline, 0.02 "excessive" ablation | not present | β = 0.001 |
| **Teacher encoder** | `E^t(x^t)`: contact forces, base height, full body state; MLP [256,128] | `modules/encoder.py::TeacherEncoder`, input = wheel contact forces (6) + 0.4·base height (1) + root state (13); latent 12 | MLP [256,128] → 12. Input: foot contact forces, torso height, root quat / velocities, all body heights, assist wrench |
| **Student encoder** | `E^s(o_{t:t−H})`, MLP [256,128]; MSE to teacher latent over **all** trajectories | `StudentEncoder`; H = `add_time_number` = 5; `algorithms/encoder_mse.py`: lr 1e-3, 5 epochs, 4 mini-batches, clip 0.8, all envs | Same: H = 5, same optimizer settings, all envs |
| **Observation history** | `o_{t:t−H}` | deque of 5 noisy `obs_buf` | mjlab observation group with `history_length=5` |
| **Teacher/student split** | 3000 teacher + 1000 student of 4000 | `num_envs_teacher = 3000`, teacher = env index < 3000 | Same: index < 3000 of 4000 |
| **Latent routing ("or" module)** | teacher rows use `z^t`, student rows `z^s` | rollout: correct. **update**: overwrites `obs_batch[:num_teacher]` after shuffling, so the rows that get a fresh teacher latent are random, not the teacher envs (bug) | Per-sample teacher mask stored in the rollout; fresh `z^t` (with grad) only on teacher rows, stored detached `z^s` on student rows (paper: student group updates `θ` only) |
| **Critic input** | `z_t ⊕ s_t`, `s_t ∈ ℝ^188` (height map + base height) | `cat(z^t, privileged_obs)` for all envs | `z^t ⊕ s_t`. Flat ground, so no height map: `s_t` = noise-free `o_t` + base linear velocity + torso height |
| **Actor / critic** | [512,256,128] each | same, ELU | same, ELU |
| **Stage-wise rewards** | Stages r_u → r_s → r_w; switch when `|S_1| > 2/3 N` (`h > h1`) and `|S_2| > 2/3 N` (`h > h2`); targets h1, h2, h3 | `envs/bipedal_wheeled/env.py::_reward_update`, re-evaluated **every step** (non-monotonic): group 2 by default, group 3 when 2/3 are above 0.4 m, group 4 when 2/3 also have `force_scale < 100`. `reward_group_1` is never selected. Per-env target height from the batch mean (`_update_base_height_target`) | Text semantics, re-evaluated from the population every env step like Alg. 1 and the code (non-monotonic; the stage can regress). Alg. 1's `p` formula maps p=1 to `r^w` and p=2 to `r^s`, which contradicts the text; text followed |
| **Stage heights** | not numeric in the paper | `[0.3, 0.45, 0.6, 0.75]` of 0.75 m; thresholds 0.35 / 0.4 | Mini-Pi geometry: h1 = 0.19 m (lowest balanced crouch, 0.55 × stance), h2 = 0.30 m (0.87 × stance), h3 = 0.335 m (walking height). §5 has the derivation |
| **Reward table** | Table II (weights per stage) | `config.py::reward_group_1..4`; functions in `common/reward_functions.py`; weights × dt (0.02) | Same weights where the term carries over; wheel-specific terms replaced (§4). dt scaling identical (mjlab `reward = raw · w · step_dt`, step_dt = 0.02) |
| **Termination penalty** | on ground > 10 s → terminate + penalty | `using_time_overstep_terminate = False` in released configs | Paper version: torso below 0.15 m for > 10 s → terminate, weight per stage |
| **8000 iterations** | 8k (humanoid comparison); wheeled assist removed by 3k | `max_iterations = 1500` default | 8000 |
| **Locomotion pretraining** | init from a policy trained with `r^w` until elementary walking (~200 it) | not in the released code | Separate walk-only task with the same network/obs layout, ~200 iterations, then weights-only init |
| **Domain randomization** | Table III | `config.py::domain_rand`: friction [0.1,1.2], added mass [-0.1,1.2], restitution, CoM offsets; PD/motor-strength rand **off**; velocity pushes after 2000 it | §6 |
| **Timing** | sim 200 Hz, policy 50 Hz | dt 0.005, decimation 4 | Mini-Pi: 500 Hz physics (2 ms), decimation 10, policy 50 Hz. Unchanged: the 5 ms step chatters on Mini-Pi |
| **PPO** | clip 0.2, γ 0.99, λ 0.95, lr 1e-3 adaptive, KL 0.01, 5 epochs, 4 mini-batches, entropy 0.01 | same | same |
| **Actions** | `q_target = q_default + scale·a` (absolute), scale 0.2–0.6 | absolute; actions clipped at ±50 | Mini-Pi uses **relative** targets (`q_target = q + 0.13·a`, clamped to joint limits), as in the existing Mini-Pi get-up. With absolute targets at 0.13 rad, getting up needs `|a| ≈ 12`. Relative targets bound the per-step P-torque to `kp·0.13·|a|` |
| **Torque** | not discussed | effort limits from URDF | Operational envelope 9 Nm for the FTSR task only (asset stays 16 Nm) |

### Other reference quirks noted

- `TSPPO.update` uses `mini_batch_generator_together`, which shuffles the samples. The
  teacher-latent injection assumes the first `num_teacher` rows are teacher envs,
  which no longer holds after the shuffle.
- `student_encoder_inference_decimation = 10` is in the config but never used.
- `pen_base_orientation_l2` isn't `‖g_xy‖`: it is `0.02‖g_xy‖² + 2(g_z+1)² − clamp(exp(−‖g_xy‖²/0.25),0,0.1)`.
  The reproduction uses the paper's `‖g_xy‖₂` plus an upside-down term (released
  `pen_base_orientation_z_l2`), because `‖g_xy‖` is 0 when lying flat on the back.
- The released reward groups are 4 (`reward_group_1..4`), and their weights differ
  from Table II. Table II is used.

## 4. Reward-term mapping (paper Table II → Mini-Pi)

| Paper term | Mini-Pi term | Note |
|---|---|---|
| Lin. vel. `exp(−8.3‖v_xy − v_cmd‖²)` | `track_lin_vel` with std = 1/√8.3 | gated on torso > 0.8 × stance |
| Ang. vel. `exp(−8.3‖ω_z − ω_cmd‖²)` | `track_ang_vel` | same gate |
| Orient. `‖g_xy‖₂` | `orientation_l2norm` + upside-down indicator | see §3 quirks |
| Torques `‖τ‖²` | `joint_torques_l2` | |
| DOF acc. | `joint_acc_l2` | |
| DOF vel. | `joint_vel_l2` | |
| Act. rate / Act. smooth | `action_rate_l2` / `action_acc_l2` | |
| Dof pos. `‖q − q_nominal‖²` | `joint_deviation_l2` (q_nominal = 0, the stance) | |
| Dof ener. `‖q̇τ‖²` | `joint_power_l2` | |
| Base heig. `exp(−8.3‖h − h_cmd_i‖²)` | `stage_height_tracking` | `h_cmd_i` set by the stage |
| Termin. | `is_terminated` + `low_for_too_long` termination | |
| Feet dist. (wheels 0.35 m) | `feet_lateral_distance` | Mini-Pi stance width = 0.163 m |
| Leg bias `‖q_left − q_right‖` | `leg_mirror_error` (mirrored roll/yaw axes) | |
| No fly (both wheels off ground) | `no_feet_contact` | from the foot contact sensor |
| Wheel force `‖f_wheel − M‖` | `feet_support_force`: `exp(−|ΣF_z,feet − m g| · 0.05 · 68/68)` | "leg usage": feet carry the weight |

Walking-only additions for a legged (not wheeled) robot: `feet_air_time` and
`feet_slip` from mjlab's velocity task, at small weights.

## 5. Stage heights for Mini-Pi

The paper gives no numbers. The released code uses fractions of the 0.75 m stance
(0.4 / 0.6 / 0.8 / 1.0). Mini-Pi has a measured stance of 0.345 m, and these are its
geometric landmarks:

- lying 0.085 m; sitting ≈ 0.06–0.12 m;
- lowest balanced crouch 0.19 m (0.55 × stance) → **h1 = 0.19 m**, the "upper body
  erect, feet under the body" state that ends r_u;
- **h2 = 0.30 m** (0.87 × stance), nearly straight legs: `S_2` must be reachable by
  more than 2/3 of the envs, so it sits clearly below the stance;
- **h3 = 0.335 m**, the walking height (slight knee bend), target of r_w.

## 6. Domain randomization (paper Table III → Mini-Pi)

| Paper | Mini-Pi | Reason |
|---|---|---|
| base mass U(−0.1, 1.2) kg | torso added mass U(−0.1, 0.5) kg | ≈ −1.5 % … +7 % of 6.94 kg, the same relative order as on the larger wheeled robot |
| CoM x ±0.02, y ±0.01, z ±0.02 m | x ±0.015, y ±0.01, z ±0.015 m | torso is ~0.1 m tall |
| friction U(0.1, 1.7) | foot/body friction U(0.3, 1.5) | MuJoCo resolves friction by geom priority / max, not PhysX averaging. 0.1 is ice for the feet |
| restitution U(0.3, 1.0) | not applied | MuJoCo has no restitution coefficient (soft contacts via solref) |
| kp/kd × U(0.95, 1.05) | `dr.pd_gains` scale U(0.95, 1.05) | dimensionless, kept |
| motor strength U(0.85, 1.05) | `dr.effort_limits` scale U(0.85, 1.05) on the 9 Nm envelope | dimensionless, kept |
| initial joint angle U(0.5, 1.5) × default | additive U(−0.3, 0.3) rad, clamped to limits | Mini-Pi default angles are 0, so a multiplicative range does nothing |
| initial base Euler U(−0.3, 0.3) | roll/pitch U(−0.3, 0.3) rad around each fallen pose, yaw uniform | kept |
| 4 fallen states (front, back, two sides) | supine, prone, left side, right side, dropped from 0.15 m and settled 0.5 s with actions off | as paper |

## 7. Implementation plan (from the code above)

1. New package area `src/minipi_getup/ftsr/`. The existing `getup/` package is
   untouched apart from the shared registration import.
   - `mdp/`: assistance (Eq. 4 event + cost observation), stage manager (population
     2/3 rule), rewards, observations, events (multi-pose fallen reset), safety
     metrics.
   - `config/`: `Mjlab-FTSR-MiniPi` (full FTSR), `Mjlab-FTSR-MiniPi-Walk` (r_w
     pretraining). Separate robot cfg with 9 Nm effort limit and relative action
     scale 0.13.
   - `rl/`: native FTSR runner (teacher/student encoders, shared actor-critic, cost
     critic, constrained GAE / Eq. 8, student MSE, TensorBoard logging, checkpoint
     I/O, ONNX export). Plugged in through `register_mjlab_task(runner_cls=...)`, so
     `uv run train` works unchanged.
2. Evaluation script: fixed seeds × {supine, prone, left, right} × velocity commands,
   writes `logs/ftsr_analysis/results.csv`.
3. Phases A (validation) → B (short diagnostics) → C (pretrain) → D (8000 it), then
   tuning (at most 3 rounds), selection, docs and local commits. Nothing is pushed.
