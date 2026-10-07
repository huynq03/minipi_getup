# HoST Mini-Pi (`pi_ground`) → MuJoCo/MJLab port audit

Task id `Mjlab-Getup-HoST-MiniPi`. Code: `src/minipi_getup/host/` (env, config, asset,
validation, evaluation, rendering) and `src/minipi_getup/host_rl/` (HoST's RSL-RL fork).
Reference: read-only checkout `/home/huy/HoST` (untouched).

Experimental question: with HoST's algorithm, rewards, curricula, observations, action
semantics, PPO changes, networks and hyperparameters kept fixed, and only Isaac Gym /
legged_gym replaced by MuJoCo / MJLab, does Mini-Pi learn to stand up?

Ground truth was the source code, not the paper or README. Files read:
`envs/pi/pi_config_ground.py`, `envs/pi/pi_host_ground.py`, `envs/base/legged_robot_config.py`,
`envs/base/base_config.py`, `envs/base/base_task.py`, `envs/g1/g1_utils.py` (`tolerance`),
`envs/__init__.py` (registration), `scripts/train.py`, `scripts/play.py`,
`scripts/eval/eval_ground.py`, `utils/task_registry.py`, `rsl_rl/modules/actor_critic.py`,
`rsl_rl/algorithms/ppo.py`, `rsl_rl/storage/rollout_storage.py`,
`rsl_rl/runners/on_policy_runner.py`, `resources/robots/pi_12dof/urdf/pi_12dof_release_v1.urdf`.

## 1. How the port is built

The release's environment (`LeggedRobot_Pi`) is transcribed method by method into
`host/env.py` and keeps its buffer names, call order and formulas. It runs on MJLab's
`Scene`, `Simulation` and `Entity` primitives. The port does not use MJLab's
manager-based env, because that env's fixed step order (curriculum before reset, rewards
scaled by dt, a single reward vector, observation history cleared at reset, no hook
between physics substeps) cannot express HoST's semantics. `PiCfg` and `PiCfgPPO` are
copied into `host/config.py` with the same class inheritance. HoST's `ActorCritic`, `PPO`
and `RolloutStorage` are copied verbatim into `host_rl/`. The runner is copied with
logging- and diagnostics-only additions, marked `PORT:`.

## 2. Release audit (what HoST actually does)

| HoST concept | Source (file :: function) | Exact formula / behavior | Constants | Isaac API / data | MJLab equivalent | Porting risk |
|---|---|---|---|---|---|---|
| Timing | `pi_config_ground` sim/control; `_parse_cfg` | `dt = decimation * sim.dt`; `max_episode_length = ceil(10/dt)`; time-out when `episode_length_buf > 500`, so 501 steps | sim.dt 0.005, decimation 4, dt 0.02, 10 s | `gym.simulate` | `Simulation.step` | low |
| Unactuated period | `__init__`, `step`, `compute_observations`, `check_termination` | `unactuated_time = 30 * 0.02/dt = 30`. Each substep: `actions *= real_ep_len > 30` (counter before the increment). Obs `*= real_ep_len > 30` (after the increment). Velocity terminations gated the same way | 30 steps = 0.6 s | – | same code | low |
| Action → torque | `_compute_torques` | `target = q + delay(a * s)`; `tau = Kp*Kpf*(target - q) - Kd*Kdf*qd`; `tau = ms*tau + offset`; `clip(±effort)`. Recomputed every 5 ms substep from the current `q`, so the P term is `Kp*a*s` and never sees a position error | Kp hip_pitch 30, hip_roll 15, thigh 15, calf 30, ankle_pitch 12, ankle_roll 5; Kd 0.2 all; effort 20 N m (URDF) | `set_dof_actuation_force_tensor` (effort mode) | motor actuators, `set_joint_effort_target` per 5 ms | low |
| Action rescale | `_init_buffers`, `update_force_curriculum` | per-env `s` starts at `control.action_scale = 1`; `s -= 0.02` (clamp ≥ 0.25) for the reset batch when the curriculum fires | 1.0 → 0.25 | – | same code | low |
| Delay | `_compute_torques`, `reset_idx` | buffer of 5 *substep* entries, pushed every 5 ms; `target = q + buf[delay_idx]`, `delay_idx ∈ {0..4}` gives 4..0 substeps (0–20 ms). Zeroed for reset envs | max 5 | – | same code | low |
| Pull force | `step` | after each `simulate`: `F[:, base, 2] = force`, `*= real_ep_len > 30`, `*= proj_g_z < -0.8`, then `apply_rigid_body_force_tensors(F)`. Default space **ENV_SPACE** (world axes, at body COM), so it acts during the *next* simulate | 15 N on `base_link` (the only body matching `base_link`) | `apply_rigid_body_force_tensors` | `xfrc_applied` written before the next 5 ms | medium (API frame verified in the Isaac Gym docs) |
| Force curriculum | `update_force_curriculum` (called from `reset_idx` after the state reset) | `if mean(old_headheight[env_ids]) > 0.37: force[ids] = (force - 20).clamp(0); s[ids] = (s - 0.02).clamp(0.25)`. A single success takes the force from 15 N to 0 | threshold 0.37 | – | same code | low |
| Observations | `compute_observations` | `[w_b*0.25, g_b, q, qd*0.05, actions, s + U(±0.025)]` = 43; noise `U(±1)*vec`; masked; history `obs = cat(obs[:, 43:258], cur)`. **Not cleared at reset** | noise ang_vel 0.05, gravity 0.05, q 0.01, qd 0.075, actions/s 0; clip ±100 | – | same code | low |
| Critic | `num_privileged_obs = None` | the critic sees the same 258-dim actor obs | – | – | same | low |
| Task reward | `compute_reward` | `r_task = 1 * orientation * head_height` (product, no dt) | – | – | same | low |
| orientation | `_reward_orientation` | `tolerance(-g_z, [0.99, inf], margin 1, value_at_margin 0.05)`, Gaussian sigmoid | – | – | same | low |
| head_height | `_reward_head_height` | `h = z(keyframe_head_link) - mean z(ankle_roll links)`; `tolerance(h, [0.37, inf], 0.37, 0.1)`; stores `old_headheight = h` | head = base +0.08 m (fixed link) | rigid_body_states | `body_link_pos_w` | low |
| Constraint groups | `_prepare_reward_function`, `compute_reward` | group = name prefix; `rew = f() * scale * dt`, summed into its group; `only_positive_rewards` = base `False` | weights [2.5, 0.1, 1, 1] | – | same | low |
| regu terms | `_reward_*` | dof_acc `Σ((qd_prev - qd)/dt)²`; action_rate; smoothness `Σ(a - 2a₁ + a₂)²`; torques `Στ²` (last substep); joint_power `Σ|qd||τ|`; dof_vel; tracking `Σ(target - q)²` (last substep target); pos limits (0.9 × URDF, both ends multiplied); vel limits `Σclip(|qd| - 0.9*5, 0, 1)` | -2.5e-7, -0.01, -0.01, -2.5e-6, -2.5e-5, -1e-3, -2.5e-4, -100, -1 | dof_props | URDF values | low |
| style terms | `_reward_*` | hip_yaw (thigh joints) and hip_roll: `max|q|>1.4 OR min|q|>0.9`; knee: `max|q|>1.65 OR min q<-0.6`; foot displacement: `exp(-2*clamp(d²_xy(base, ankle_pitch), 0.3))*(foot z<0.15)*(base z>0.34)`; ground_parallel: mean over feet of `var(10*z)` over 5 ankle bodies (roll link + 4 auxiliaries) < 0.05; feet_distance `‖l - r ankle_pitch‖ > 0.45`; style_ang_vel_xy `exp(-2‖w_xy‖²)*(z>0.25)` | -10, -10, 2.5, 2.5, -0.25, 20, -10, 1 | – | same | low |
| target terms | `_reward_*` | all `*(base z > 0.34)`: ang_vel_xy `exp(-2‖w_xy‖²)`; lin_vel_xy `exp(-5‖v_xy‖²)` (body-frame COM velocity); feet_height_var `exp(-2*clamp(|10 z_l - 10 z_r|, 0.2))`; target_orientation `exp(-5‖g_xy‖²)`; target_base_height `exp(-20|z - 0.34|)` | 10, 10, 2.5, 10, 10 | – | same | low |
| Terminations | `check_termination` | time-out; `abs(max_j qd) > 300` (signed max, then abs); `‖v_b‖ > 20`; both after the unactuated period; no contact terminations | 300, 20 | – | same | low |
| Reset | `reset_idx`, `_reset_dofs`, `_reset_root_states` | `q = clip(0*U(0.9,1.1) + U(±0.1), 0.9*URDF limits)`, `qd = 0`; root = init (pos [0,0,0.351], quat xyzw [0,-1,0,1] unnormalized, zero velocity); no xy offset (`custom_origins` False) | – | `set_*_state_tensor_indexed` | `write_joint_state`/`write_root_state` | medium (quaternion) |
| DR (resampled every reset) | `reset_idx`, `_init_buffers` | Kp/Kd factors U(0.85, 1.15); motor strength U(0.9, 1.1); actuation offset U(±0.05)*20 N m; delay_idx | – | – | same code | low |
| DR (once, at creation) | `_process_rigid_shape_props`, `_process_rigid_body_props` | friction U(0.1, 1) and restitution U(0, 1) per env (all robot shapes); payload U(-2, 3) kg on the torso, **then overwritten** because link-mass scaling U(0.8, 1.2) applies to every body including the torso; torso COM += U(±0.03)*[4, 4, 2]; `recomputeInertia=True` | – | PhysX shape/body props | `geom_friction`, `body_mass`, `body_inertia`, `body_ipos` per world | medium |
| Pushes | `_post_physics_step_callback` | **disabled** (the call is commented out) although `push_robots = True` | – | – | none | – |
| Episode log | `reset_idx` | `rew_<term> = mean(sum)/10 s`; group values; force mean; action scale | – | – | same | – |
| ActorCritic | `actor_critic.py` | actor 258→512→256→128→12 ELU + **tanh**; 4 critics 258→512→256→1 ELU; `std` raw parameter (not log), init 0.8 | – | – | copied | low |
| Storage / GAE | `rollout_storage.py::compute_returns` | per-group GAE (`[T, N, 4]`); each group's advantages normalized over all T×N; `A = Σ_g w_g * Â_g` | λ 0.95, γ 0.99 | – | copied | low |
| Transition filter | `ppo.py::process_env_step` | a step is stored only if `norm(obs) > 1e-4` over the whole batch, so the first iteration (all envs unactuated together) stores 20/50 and the rest of the buffer stays stale/zero | – | – | copied | low |
| Time-out bootstrap | `ppo.py::process_env_step` | `r += γ * V * time_outs`. `extras["time_outs"]` is only refreshed on steps that reset some env, otherwise it is stale | – | – | copied (staleness kept) | low |
| Minibatches | `mini_batch_generator` | uses steps `[:-1]` (49 of 50) with next obs; 4 minibatches × 5 epochs | – | – | copied | low |
| Smoothness loss | `ppo.py::update` | `eps = lb/(ub - lb)`; `c_pi = ub*eps = 0.111`; `c_v = 0.1*c_pi`; `mix = obs + cont*U(±1)*(next - obs)`; `L += c_pi*‖mu - pi(mix)‖² + c_v*‖V - V(mix)‖²` | ub 1, lb 0.1, value coef 0.1 | – | copied | low |
| PPO | `ppo.py`, `PiCfgPPO` | clip 0.2, γ 0.99, λ 0.95, value coef 1, clipped value loss, entropy 0.01, 5 epochs, 4 minibatches, lr 1e-3 adaptive (KL 0.01, ×/÷1.5, [1e-5, 1e-2]), grad norm 1 | – | – | copied | low |
| Runner | `on_policy_runner.py`, `train.py` | 50 steps/env/iteration; 12000 iterations; `init_at_random_ep_len` randomizes only `episode_length_buf`; save every 100; seed 1; `env.reset()` = reset all + one zero step | – | – | copied | low |
| Eval | `scripts/eval/eval_ground.py` | `pull_force = False`, no noise, `action_scale = 0.25`, 5 s episodes, 5 episodes, DR as configured; success = (base ever > 0.7 m) and not (base < 0.5 m afterwards); G1 thresholds | – | – | `host/evaluate.py` (scaled) | medium |

## 3. Component comparison (HoST vs this port)

| Component | Original HoST | MJLab implementation | Exact match? | Difference | Reason | Expected behavioral impact |
|---|---|---|---|---|---|---|
| Robot asset | `pi_12dof_release_v1.urdf` (6.92 kg, 20 N m, 5 rad/s) | `host/assets/pi_12dof_host.xml`, generated mechanically from the same URDF; meshes byte-identical | Kinematics, inertials, limits and collision shapes: yes | The project's own `cl_pai.xml` is a **different revision** (knee/ankle bake 0.65/-0.4 vs 0.55/-0.3 rad, much narrower joint ranges, capsule collisions, 16 N m) and is **not** used | Fidelity to HoST | none vs HoST; results do not transfer to `cl_pai.xml` without a new experiment |
| Joint order | Isaac DOF order from the URDF tree | MJCF order r_hip_pitch … r_ankle_roll, l_hip_pitch … l_ankle_roll | Probably (same tree order); irrelevant | Every reward and gain looks up joints by name | – | none (policy trained from scratch) |
| Massless links | `keyframe_head_link`, 8 `auxiliary_*` (`dont_collapse`) | massless welded bodies | yes (frame origins) | Isaac may give them a tiny default mass | – | negligible |
| Initial state | pos (0,0,0.351), quat xyzw (0,-1,0,1) **unnormalized**; supine | wxyz (0.7071, 0, -0.7071, 0); projected gravity at reset (-1, 0, 0) = supine (validated, renders in `results/host/validation/`) | yes (normalized) | PhysX needs unit quaternions; the intended pose is the normalized one | – | none |
| Joint reset | `clip(0*U + U(±0.1), 0.9*limits)`, qd 0 | same | yes | – | – | – |
| Physics timestep | PhysX TGS, 5 ms, 1 substep, 8 position / 1 velocity iterations | MuJoCo (mujoco_warp) implicitfast + Newton; **2 × 2.5 ms per 5 ms** control substep | **No** | torque, delay buffer and pull force are still computed once per 5 ms and held for both 2.5 ms steps (zero-order hold like PhysX) | A single 5 ms MuJoCo step **diverges** for these light, 20 N m-driven links: under the random initial policy 0.25 per 1k env-steps hit base speed > 20 m/s or qd > 300 rad/s (up to 5e4 rad/s), and 0 with 2.5 / 1.67 / 1 ms (peak qd converges at ~125 rad/s) | Removes a MuJoCo-specific instability; the control semantics are unchanged |
| Policy timing | 50 Hz, decimation 4, 10 s + 1 step | same | yes | – | – | – |
| Contact model | PhysX: rigid contacts, contact offset 0.01, rest offset 0, max depenetration 1 m/s, bounce threshold 0.5 m/s, average friction combine | MuJoCo soft contacts (default solref/solimp), pyramidal cone, condim 3 | **No** | different contact dynamics; no depenetration velocity cap | simulator | moderate: contact-rich get-up depends on it |
| Self-collision | enabled (`self_collisions = 0`), adjacent links excluded | enabled, parent–child excluded | yes | – | – | – |
| Friction | robot shape μ_r ~ U(0.1, 1) per env; plane static 0.8 / dynamic 0.7; PhysX average combine | robot geoms get priority with μ = (μ_r + 0.75)/2; one coefficient for static and dynamic | approx. | MuJoCo has no separate static/dynamic coefficients | simulator | small |
| Restitution | robot U(0, 1), plane 0.3, average | **not reproducible**: MuJoCo contacts are not restitution-based. Sampled and ignored | **No** | – | simulator | small–moderate (bounce on landing) |
| Body damping | linear/angular damping 0.01 on every link | none | **No** | – | no MuJoCo equivalent | negligible |
| Armature | 0.01 on every DOF (asset option) | joint armature 0.01 | yes | – | – | – |
| Joint velocity limit | URDF `velocity = 5`; whether PhysX enforces it in effort mode is **undocumented**. PhysX articulations also cap joint speed by default (`maxJointVelocity`) | **no cap** (MuJoCo has none) | **Unknown** | joints reach ~125 rad/s under exploration | simulator | **potentially large**: if Isaac clamps at 5 rad/s (or 100), HoST's motions were much slower than here. Logged (`Robot/joint_vel_abs_*`, eval peaks) |
| PD / torque | manual PD each 5 ms, clip ±20 | identical code, motor actuator with ctrlrange ±20 | yes | – | – | – |
| Action definition | `target = q + delay(a*s)` | identical | yes (validated: 0.1 → 0.1 rad at s=1, 0.025 rad at s=0.25) | – | – | – |
| Action-scale curriculum | -0.02 per triggering reset batch, ≥ 0.25 | identical | yes | – | – | – |
| Pull force | 15 N, world +Z at base COM, gated (real_ep_len > 30, g_z < -0.8), applied in the next simulate | identical; `xfrc_applied` on `base_link` for the next 5 ms | yes (validated: Δa_COM = 2.1666 vs 15/M = 2.1672 m/s²; zero when lying or unactuated) | – | – | – |
| Observations | 43 × 6 = 258; noise; masking; history not cleared at reset | identical | yes (validated order, scales, noise vector, shift) | `base_lin_vel` uses the root **COM** velocity (PhysX root velocity is the COM velocity) | – | none |
| Rewards | 2 task (product) + 22 constraints in 3 groups, ×dt | identical code; per-term values printed for lying and standing (`validation.json`) | yes | – | – | – |
| Reward grouping | `rew_buf [N, 4]` | identical | yes | – | – | – |
| Domain randomization | see §2 | identical for Kp, Kd, motor strength, offset, delay, initial q, link mass (+ payload overwrite), COM; friction approximate; restitution impossible; pushes off in both | mostly | inertia rescaled ∝ mass to mirror `recomputeInertia=True` | – | small |
| Settled pose under DR | – | 1024-env study: without DR 100% supine after 0.6 s; with full HoST DR 36% supine, 17% already sit upright (inside the force gate), 10% on the side, mainly from the ±1 N m actuation offset acting during the passive window | follows from the release code | – | – | the start-state distribution is broader than "supine" |
| Termination | time-out, qd > 300, ‖v‖ > 20 | identical | yes | – | – | – |
| PPO | HoST fork | copied verbatim; additions only observe (stats, non-finite checks, failure dump) | yes | – | – | – |
| Multi-critic | 4 critics, per-group GAE and normalization, weighted sum | copied | yes (validated shapes [T,N,4] → [T,N]; per-group mean 0 / std 1) | – | – | – |
| Advantage calc | as in §2, incl. first-iteration partial storage (KL = inf → lr floor) | copied; reproduced (validation: 20/50 filled, KL inf) | yes | – | – | – |
| Smoothness loss | c_pi = 0.111, c_v = 0.0111 | copied | yes | – | – | – |
| Training schedule | 4096 envs × 50 steps, 12000 iterations, save 100, seed 1 | same; 4096 envs fit (18.4 GB on an RTX 3090) | yes | – | – | – |
| Checkpoint `iter` | saves `current_learning_iteration` (the *start* iteration) | also saves the real `it` and the adaptive lr (resume only) | – | – | resume correctness | none on training |
| Evaluation | G1 thresholds 0.7/0.5 m on base height | Mini-Pi: ×(0.34/0.75): stand 0.317 m, fall 0.227 m; counted only once actuated (Mini-Pi resets at 0.351 m > 0.317); plus a stricter "sustained" criterion (base > 0.317 and g_z < -0.8 for the whole last 1 s) | adapted | the release has no Mini-Pi eval | – | – |

## 4. Places where exact equivalence is impossible

1. **Contact solver.** PhysX TGS rigid contacts vs MuJoCo soft, convex, complementarity-free
   contacts: no depenetration velocity cap, and no separate static/dynamic friction.
2. **Restitution.** Not representable in MuJoCo; randomized values are ignored.
3. **Integration step.** MuJoCo needs 2.5 ms steps where PhysX used 5 ms. HoST's control
   semantics are kept by holding torque and force for 5 ms.
4. **Joint velocity limit.** Unknown in Isaac Gym effort mode; MuJoCo has none.
5. **Link velocity damping (0.01)** and Isaac's handling of massless links: not modeled.
6. **`recomputeInertia=True`.** Approximated by scaling the inertia with the mass.

## 5. Validation (`python -m minipi_getup.host.validate`, `results/host/validation/`)

All 12 checks pass:

- **timing:** 5 ms control substep, 2 × 2.5 ms MuJoCo, decimation 4, 0.02 s, 500 + 1 steps, 30 unactuated steps.
- **unactuated period:** zero obs and zero actions for 30 steps.
- **first actuation:** obs non-zero after step 31; first non-zero action in step 32 (0.62 s), as in the release.
- **action semantics:** `q_target - q` = 0.1 at s = 1 and 0.025 at s = 0.25; torque error < 1e-6.
- **delay buffer:** indices 0..4 give 4..0 substeps of delay.
- **observation layout:** 43 × 6 = 258, order, scales and history shift.
- **noise vector:** values as listed in §2.
- **pull force:** 15 N world +Z on `base_link`; COM acceleration matches F/M to 0.03%; zero when lying or unactuated.
- **rewards:** finite per-term values for lying and standing states. Standing at the zero pose gives head − feet 0.382 m, base 0.349 m and task reward 1.0.
- **multi-critic PPO:** shapes and normalization as in §2; an update runs.
- **reset pose:** supine.

The constraint-buffer overflow seen at first (`nefc overflow`) was a port bug. It is
fixed with `njmax = 1200` and `nconmax = 300`; no overflow appears in training logs.
