# FTSR reference audit v2: getup_gym release vs paper, ported to Mini-Pi

Written 2026-10-07 on branch `rebuild/ftsr-reference-minipi`, before any code.

Sources and authority (algorithmic):

1. `/home/huy/getup_gym` executable code (HEAD `238d296`, read-only, untouched).
2. The paper (Hou et al., RA-L 2026, arXiv:2606.14270). No PDF is on this machine; the
   LaTeX source the user supplied at the start of the project (session transcript) was
   used. Equation numbers below follow that source.
3. The archived experiments (`archive/ftsr-experiments-20261007`, HEAD `0a8cb71`):
   evidence only.

Paths are relative to `/home/huy/getup_gym/getup_gym/` unless stated. Every statement
was traced from assignments and call order, not from comments or names.

Deviation classes used: `SIMULATOR_PORT`, `ROBOT_ADAPTATION`, `HARDWARE_CONSTRAINT`,
`REFERENCE_BUG_FIX`, `PAPER_COMPLETION`. Points the paper leaves open and that are not
yet derived are marked `PAPER_AMBIGUITY`; values to be calibrated are marked
`UNRESOLVED`.

## 1. Executable path actually used by `scripts/train.py`

`scripts/train.py` → `utils/registry.py` (`BipedalWheeled` = `JROwheel`, `GetupCfg`,
`GetUpPPO`) → `common/task_registry.make_alg_runner` → **`runners/on_policy_runner.py`
`OnPolicyRunner`** → `algorithms/ts_ppo.py` `TSPPO` → `storage/rollout_storage.py`
`RolloutStorage`; student via `algorithms/encoder_mse.py` `EncoderMSE`.
`runner.learn(max_iterations, init_at_random_ep_len=True)`.

The `HhdRslRl/Runner/TSon_policy_runner.py` path (which indexes `extras["force"]`
directly and would raise `KeyError`) is **not** used by `train.py`.

### One environment step (`envs/legged_robot.py::step`, `post_physics_step`)

1. `actions = clip(actions, ±50)` (`clip_actions = 50`).
2. For each of `decimation = 4` physics steps (`sim.dt = 0.005`, so the policy runs at
   50 Hz and physics at 200 Hz):
   - If `1 < it < 3000` (it = `(common_step_counter-1)/24 - 1`):
     `_base_force_pull_up()` (§4).
   - `actions *= (real_episode_length_buf > unactuated_time)`; actions are **zeroed**
     for the first `unactuated_timesteps = 30` policy steps = **0.6 s**
     (`unactuated_time = 30·0.02/dt` sim steps compared against a counter that counts
     policy steps; see quirk Q6). A zero action means target = default pose with the PD
     active, i.e. "hold default pose", not a passive robot.
   - `_compute_torques` (bipedal env): explicit PD in torch,
     `tau = kp·(scale·a + q_default − q) − kd·qdot`, clipped to the URDF effort (120 Nm
     legs). Wheels are velocity controlled.
3. `post_physics_step`: counters, base state, `_pre_reward_callback`,
   `_update_base_height_target` (§5), `check_termination`, `_reward_update` (§5),
   rewards via `RewardManager.compute_reward(env, reward_group)`, `reset_idx`,
   `compute_observations`, then the `last_*` buffers, then `last_observations_save()`.
4. Observations clipped to ±100.

### Rollout (`OnPolicyRunner.learn`)

```
obs_t' = [z, o]  with z = E_t(x^t) for rows [0:3000], E_s(o^s) for rows [3000:4000]
critic_t' = [E_t(x^t), privileged_obs]                    (all rows)
act(obs_t', critic_t') ; z_t, z_s recomputed from the CURRENT teacher/student obs
encoder storage <- (z_t, o^s, x^t)                       (all rows)
env.step -> o_{t+1}; obs = [z (from step t), o_{t+1}]      <- latent lags one step (Q3)
process_env_step: r += gamma·V·time_out (time-out bootstrap)
```

Then `compute_returns(critic_obs, force_buf)`, `alg.update()`,
`student_encoder_alg.update()`, save every 50 iterations.

## 2. Verified parameters

| Item | Release value | Where |
|---|---|---|
| Envs / teacher / student | 4000 / 3000 / 1000 (rows `[0:3000]` teacher) | `config.env.num_envs`, `num_envs_teacher` |
| Latent | 12 (`layer_size`) | `config.env` |
| Actor / critic | [512, 256, 128], ELU, `init_noise_std 1.0`, state-independent std | `modules/actor_critic.py` |
| Teacher / student encoder | MLP [256, 128] → 12, ELU, linear output | `modules/encoder.py` |
| PPO | clip 0.2, γ 0.99, λ 0.95, entropy 0.01, lr 1e-3 adaptive (KL 0.01, ×/÷1.5, [1e-5, 1e-2]), 5 epochs, 4 minibatches, max grad 1.0, value coef 1.0, clipped value loss | `GetUpPPO.runner.algorithm_config` |
| Steps per env | 24 | `num_steps_per_env` |
| Max iterations | **1500** in the config (the paper reports 8000) | `GetUpPPO.runner` |
| Student MSE | lr 1e-3, 4 minibatches, 5 epochs, grad 0.8, `MSE(E_s(o^s), E_t(x^t).detach())`, **all 4000 envs** | `encoder_mse.py`, `student_encoder_alg_config` |
| Teacher encoder training | PPO loss only (in the PPO optimizer); `teacher_encoder_alg_type="KL-Entropy"` is not used by this runner | `ts_ppo.py` |
| Action | `q_target = q_default + scale·a`, absolute; per-joint `action_scale = [0.2, 0.6, 0.6, 1.0]` (hip roll, femur pitch, tibia pitch, wheel), clip ±50 | `env.py::_compute_torques` |
| Default pose | hip roll 0, femur 0.4, tibia −0.8 | `init_state.default_joint_angles` |
| Gains | kp 80.4 / 101.4 / 101.7, kd 3.5 / 2.5 / 2.5 (wheel velocity kp 1) | `control` |
| Timing | sim 0.005 s, decimation 4, policy 50 Hz | `sim.dt`, `control.decimation` |
| Episode | 20 s, no termination except time-out (Q8) | `episode_length_s`, `terminate_after_contacts_on=[]`, `using_time_overstep_terminate=False` |
| Unactuated start | 30 policy steps = 0.6 s; obs also zeroed during it after it 500 | `unactuated_timesteps`, `compute_observations` |
| Reward scaling | `weight · raw · dt` with dt = 0.02 | `common/reward_manager.py` |
| Assist | `base_pull_up_max = 400 N`, ends at iteration 3000 | `rewards.curriculum` |
| Robot mass | **27.669 kg** (URDF sum; torso 14.79 kg), weight 271.4 N | `robots/Owheel/urdf/robot_save.urdf` |
| Fmax / mg | **400 / 271.4 = 1.474** | computed |
| Effort / velocity limits | legs 120 Nm, 15 rad/s; wheels 40 Nm, 25 rad/s | URDF |
| Armature | 0.001 (Isaac Gym asset option) | `asset.armature` |
| Stance height | 0.75 m (`base_height_target`) | `rewards` |

## 3. Observations

| Group | Release content | Dim (JiaRan) |
|---|---|---|
| `o_t` (actor/student frame) | `base_lin_vel·2·0` (three zeros), `ang_vel·0.25`, `projected_gravity`, `commands·[2, 2, 0.25]`, leg joint pos − default (6, no wheels), `dof_vel·0.05` (8), `actions` (8) | 34 (paper: 34 ✓) |
| Noise (uniform ±) | ang vel 0.05·0.25, gravity 0.05, dof pos 0.04, dof vel 0.06·0.05; none on commands/actions | |
| Student `o^s` | the 5 most recent entries of `observations_stack`, oldest first, frame-major | 5×34 |
| Teacher `x^t` | wheel contact forces (3+3), `base_height·0.4`, full root state (13: world pos incl. env origin, quat, lin vel, ang vel) | 20 |
| Privileged obs (`privileged_obs_buf`) | `o_t` (34, with the zeroed lin-vel slot replaced by the true `base_lin_vel·2`) + 187 height samples `clip(h_root − 0.5 − h_terrain, ±1)·5` (17 × 11 grid) + base height (1) | 34 + 187 + 1 = **222** |
| Final critic input | `cat(E_t(x^t), privileged_obs)` for **all 4000 rows**, teacher latent even for student rows (`on_policy_runner.py:142,182`, `num_critic_obs += layer_size`) | 12 + 222 = **234** |
| Final actor input | `cat(z, o_t)`: `E_t(x^t)` for rows `[0:3000]`, `E_s(o^s)` for rows `[3000:4000]` | 12 + 34 = 46 |

Paper vs release for the critic (conflicts, recorded, not reconciled):

- **Critic state.** Paper Table I: Critic input `z_t, s_t` with `s_t ∈ R^188` "containing
  height map and robot base height" (187 + 1). Release: `s_t` is preceded by the full
  34-dim `o_t` (with true linear velocity), i.e. `[z_t, o_t*, s_t]` = 234, not
  `[z_t, s_t]` = 200. The paper's Fig. 2 caption says `z_t` is "concatenated with `o_t`
  and `s_t` for shared Actor-Critic networks", which is closer to the release but
  contradicts Table I.
- **Critic latent routing.** Paper Fig. 2: an "or" module selects `z^t_t` for
  teacher-group agents and `z^s_t` for student-group agents "to form `z_t` concatenated
  with `o_t` and `s_t`", which reads as the student latent reaching the critic for
  student rows. Release: the critic always gets the teacher latent (`st`) for all rows.
- Both are `PAPER_AMBIGUITY` for the port; the release behavior is the default port
  (release first), the conflict is to be revisited only if training evidence requires.

## 4. External assistance in the release (`legged_robot.py::_base_force_pull_up`)

Executed every physics step while `1 < it < 3000`:

```
h            = root_z − terrain height                     (torso origin)
time_coeff   = clamp((3000 − it_float)/3000, 0, 1)          it_float = (step_counter−1)/24
force_scale  = clamp((h_target − h)/h_target · 400 · time_coeff, 0, 400),  h_target = 0.75 FIXED
F            = [0, 0, force_scale]   on the torso, ENV (world) frame
T_x          = −10 · time_coeff · |q_y + 1| · (h_target − h)/h_target   (world x axis only)
F, T masked to 0 during the unactuated window
```

Facts: linear (not exponential) in height; fixed target 0.75 m, not the stage `h_cmd`;
the torque is a single world-x component built from the quaternion y component, which
only makes sense for the one fixed reset orientation the release uses (Q7); torque
magnitude ≈ 10·0.29·0.87 ≈ 2.5 N·m at the reset pose, at most ~20 N·m. After iteration
3000 the function is no longer called: no force, but `self.force_scale` keeps its last
value (used by the stage switch).

## 5. Stage / reward switching in the release

`_reward_update` runs every env step (stateless, population statistics, can regress):

```
group = g2                                   (default; g1 is never selected)
if   #(h < 0.35) > 2/3·N: group = g2
elif #(h ≥ 0.40) > 2/3·N: group = g3
     if #(force_scale < 100 N) > 2/3·N: group = g4
```

`_update_base_height_target` (every step): one target for all envs from the
**population mean** height `h̄`: 0.3 if `h̄ < 0.24`, 0.45 if `0.24 ≤ h̄ < 0.45`, 0.6 if
`0.36 ≤ h̄ < 0.6`, 0.75 above (thresholds `0.8·target_i`). The height target is
therefore decoupled from the reward group.

### Reward groups (weights × dt; raw formulas from `common/reward_functions.py`)

| Release term | Raw formula | g2 | g3 | g4 |
|---|---|---|---|---|
| `pen_termination` | `reset & ~time_out` (always 0, Q8) | −200 | −100 | −500 |
| `pen_base_orientation_l2` | `0.02‖g_xy‖² + 2(g_z+1)² − clamp(exp(−‖g_xy‖²/0.25), 0, 0.1)` | −2.6 | −2.0 | −15 |
| `pen_base_orientation_z_l2` | `1[−g_z < −0.2]` (upside down) | −0.6 | −1.0 | – |
| `track_base_height_exp` | `exp(−|h − h_target|/0.12)` | 4 | 6 | 5 |
| `pen_dof_acc_l2` | `Σ((q̇_{t−1} − q̇_t)/0.02)²` | −2.5e−8 | −2.5e−6 | −2.5e−6 |
| `pen_action_rate_l2` | `Σ(a_{t−1} − a_t)²` | −0.06 | −0.02 | −0.1 |
| `pen_dof_pos_bias_l2` | `(Σ_j [(q_L,j − d_j)² + |q_R,j − d_j|])² · 1[both wheels in contact]`, j = hip, femur, tibia | −0.06 | −0.14 | −0.12 |
| `pen_two_leg_bias_l2` | `Σ_j (q_L,j − q_R,j)²` (raw, not mirror-corrected) | −0.02 | −0.02 | – |
| `pen_no_fly_l2` | `1[#wheels with F_z > 1 N ≠ 2]` | −2.0 | – | −0.2 |
| `rew_wheel_contact_force` | `exp(−0.05·|ΣF_z,wheels − 10·M|)` | 4.0 | – | – |
| `pen_torques_l2` | `Στ²` | – | −1e−6 | −1e−5 |
| `track_lin_vel_xy_exp` | `exp(−‖v_xy − v_cmd‖²/0.12)·1[h > 0.6]` | – | – | 7 |
| `track_ang_vel_yaw_exp` | `exp(−(ω_z − ω_cmd)²/0.12)` | – | – | 4 |
| `pen_dof_vel_l2` | `Σ q̇²` (legs only) | – | – | −4e−3 |
| `pen_feet_distance_l2` | `clip(0.32 − d, 0, 1) + clip(d − 0.38, 0, 1)`, d = wheel xy distance | – | – | −2.0 |
| `pen_max_velocity_l2` | +1 if `v_x` within [0.8, 1.1]·cmd, −1 otherwise | – | – | +1.0 |
| `pen_dof_pos_limits` | violation beyond 0.9 × URDF range | – | – | −5.0 |
| `pen_torque_limits` | `Σ relu(|τ| − 0.8·τ_lim)` | – | – | −0.05 |
| `pen_action_smoothness_l2` | `Σ(a_t − 2a_{t−1} + a_{t−2})²` | – | – | −0.02 |
| `pen_joint_power_l2` | `Σ|q̇||τ|·1[h < 0.7]` | – | – | −1e−4 |

## 6. Force-guidance optimization in the release: inactive (proved)

Search of the whole repository (`grep -rnE "\[[\"']force[\"']\]|extras\.get\([\"']force|infos\.get\([\"']force"`):

1. **Nothing writes `extras["force"]`.** The only `extras[...]` writes in the env code
   are `"episode"` and `"time_outs"` (`legged_robot.py:219–234`).
2. The used runner reads `force_buf = self.extras.get("force")` (`on_policy_runner.py:127`)
   and `infos.get("force")` (`:168`), so `force_buf = None` for the whole run.
3. `TSPPO.act(..., force=None)` stores `transition.force = None`;
   `RolloutStorage.add_transitions` skips the copy (`transition.force is not None`), so
   `storage.force` stays all zeros. `compute_returns(..., last_force=None)` substitutes
   zeros.
4. In `RolloutStorage.compute_returns` the term `delta += force_t − γ·force_{t+1}` is
   therefore `0 − 0`. **The released force-guided optimization is a no-op.** The force
   only exists as the physical assist, so the release trains the paper's
   "Ours w/o Force Guid." ablation (Table IV(b): 97.4 % on ground).
5. Even if wired, the release form is potential-based shaping (`Φ = −F`) folded into the
   reward advantage, with no β and no torque term: different from Eq. 5–8.

Conclusion: the release cannot be the authority for the force-guided objective. The
port implements Eq. 5–8 (`PAPER_COMPLETION`, §9).

## 7. Other release quirks (traced)

| # | Quirk | Evidence | Port decision |
|---|---|---|---|
| Q1 | Teacher identity lost after shuffle: minibatch rows come from `randperm`, then `obs_batch[:num_teacher, :12] = E_t(x^t)[:num_teacher]` overwrites the first 3000 rows of each 24 000-row minibatch whatever their origin | `rollout_storage.py::mini_batch_generator_together`, `ts_ppo.py::update` | Store a teacher mask per sample; teacher rows get a fresh `E_t` latent with gradient, student rows keep their stored latent. `REFERENCE_BUG_FIX` |
| Q2 | `last_observations_save()` is called twice per step (in `compute_observations` and at the end of `post_physics_step`), so the 5-slot stack holds only 3 distinct frames: `[o_{t−2}, o_{t−1}, o_{t−1}, o_t, o_t]` | `env.py:69`, `legged_robot.py:162`, deque `maxlen=5` | H = 5 distinct consecutive frames (paper `o_{t:t−H}`). `REFERENCE_BUG_FIX` |
| Q3 | Latent lags one step: the latent concatenated with `o_{t+1}` was computed from `x_t` / `o^s_t`, and the critic gets the stale teacher latent | `on_policy_runner.py:158–182` | Compute latents from the same time step as `o_t`. `REFERENCE_BUG_FIX` |
| Q4 | Force guidance inactive | §6 | `PAPER_COMPLETION` |
| Q5 | `reward_group_1` defined but never selected | `_reward_update` | Not ported |
| Q6 | (checked, not a quirk) `unactuated_time = 30·0.02/self.dt` with `self.dt = 0.02` → 30 policy steps = 0.6 s | `env.py:27`, `legged_robot.py:1173` | Port 0.6 s |
| Q7 | All envs reset to one orientation `rot = (0, −1, 0, 1)` (normalized: −90° about y) with joint pos `default·U(0.1, 1.5)` and root velocities U(−0.5, 0.5); the paper uses four fallen poses + Euler noise | `env.py::_reset_root_states`, `_reset_dofs` | Four paper poses. `PAPER_COMPLETION` |
| Q8 | No termination except time-out: `terminate_after_contacts_on = []`, `using_time_overstep_terminate = False`, so `pen_termination` is always 0; the paper's 10 s ground termination is off | `config.py`, `check_termination` | Port the release (off) for the baseline; recorded as a tuning family candidate |
| Q9 | `pen_two_leg_bias` uses raw `q_L − q_R` although JiaRan's hip-roll limits are mirrored | `reward_functions.py:224` | Port literally (raw difference); noted |
| Q10 | `pen_dof_pos_bias` uses `(·)²` for left joints and `|·|` for right joints | `reward_functions.py:196` | Port literally; noted |
| Q11 | `total_mass` is taken from env 0 before randomization | `_process_rigid_body_props` | Use the nominal model mass |
| Q12 | Checkpoint stores actor-critic, optimizer and iteration only; encoders separately; no env step counter, no adaptive LR, so a resume restarts the assist schedule | `on_policy_runner.py::save/load` | Save everything (§9). `REFERENCE_BUG_FIX` |
| Q13 | No locomotion pretraining in the code (trains from random init; `max_iterations 1500`) | `train.py`, README | Paper Sec. III-A walking init. `PAPER_COMPLETION` |
| Q14 | During the same 0.6 s window: obs, teacher obs and student obs are zeroed during the unactuated window after it 500 | `env.py::compute_*` | Port (same window as Q6) |

## 8. Paper vs release vs Mini-Pi decision table

| Component | Paper | Release | Conflict | Chosen | Class / reason |
|---|---|---|---|---|---|
| Assist force | Eq. 4: `(1 − e^{−μ(h_cmd,i − h)})·sat(1 − t/t_tag)·F_max·n`, stage `h_cmd` | linear `(0.75 − h)/0.75`, fixed 0.75 m | yes | **Eq. 4** | `PAPER_COMPLETION`: F and T must be the CPO constraint costs (§6), and the release torque only works for its single reset pose |
| Assist torque | `T_max·log(R_tag R⁻¹)` (so(3)) | world-x scalar from `q_y` | yes | **so(3), R_tag = upright with the current yaw** (no yaw target) | `PAPER_COMPLETION` (four poses) |
| F_max | not given | 400 N = 1.474 mg | – | **1.474·m·g** of Mini-Pi (dimensionless reference scaling) = 100.4 N; ablation 1.0 mg | `ROBOT_ADAPTATION` |
| T_max | not given | world-x scalar `−10·t·|q_y+1|·(0.75−h)/0.75` | yes: the release coefficient is not the paper's `T_max` (different axis construction and argument, `|q_y+1|` ∈ [0, 2] is not a rotation angle) | **UNRESOLVED**, calibration parameter. A dimensionless reference `10/(m g L)` = 0.049 is only an order-of-magnitude anchor | – |
| μ | not given | – (the release is linear) | yes | **UNRESOLVED**, calibration parameter. Matching the release's linear factor at the lying height is one candidate, not a derivation | – |
| t_tag | step at which assist ends | iteration 3000 | no | 3000 iterations × 24 steps | release |
| Constraint optimization | Eq. 5–8, β = 0.001 | no-op | yes | Eq. 5–8 (§9) | `PAPER_COMPLETION` |
| Stages | 3 (r_u, r_s, r_w); `|S_1| > 2/3N` (h > h1) → r_s, `|S_2| > 2/3N` (h > h2) → r_w; Alg. 1's `p` formula maps p=1 to r^w, contradicting the text | 3 used groups (g2, g3, g4); g3 at 2/3 above 0.40 m; g4 also needs 2/3 with `force_scale < 100 N` (tied to the release force form); stateless every step | partly | **Release decision frequency and statelessness, paper threshold structure S1/S2**; h1, h2, h3 are **candidate** Mini-Pi parameters (§10), not fixed. Text semantics, not Alg. 1 `p` | release g4 criterion `force_scale < 100 N` depends on both height and `time_coeff`, so it is not a height threshold and cannot be mapped to a fixed h2 |
| Height target | per stage `h_cmd,i` | population-mean ladder 0.3/0.45/0.6/0.75 | yes | **per-stage `h_cmd`** (Eq. 4 needs it) | `PAPER_COMPLETION` |
| Rewards | Table II weights | groups g2/g3/g4 (§5) | yes (weights, orientation formula, kernels) | **release formulas and weights**, robot-specific replacements only | release first |
| Base height kernel | `exp(−8.3Δh²)` | `exp(−|Δh|/0.12)` | yes | release with σ scaled by stance ratio (0.12·0.345/0.75 = 0.055 m) | `ROBOT_ADAPTATION` (geometry) |
| Termination | ground > 10 s, penalty | disabled | yes | release (disabled) | release; candidate tuning |
| Teacher-student | 3000/1000, CTS, MSE on all trajectories | same + Q1, Q3 | bug | same, bugs fixed | `REFERENCE_BUG_FIX` |
| History | `o_{t:t−H}`, H not numerically stated (release 5) | 5 slots, 3 distinct (Q2) | bug | H = 5 distinct frames | `REFERENCE_BUG_FIX` |
| Initialization | pretrain with r^w until elementary walking (~200 it) | absent | yes | walking pretrain task | `PAPER_COMPLETION` |
| Resets | 4 poses + Euler U(±0.3), joints U(0.5, 1.5)·default | 1 pose, joints U(0.1, 1.5)·default | yes | 4 poses + Euler noise; joint noise additive (Mini-Pi default = 0, §10) | `PAPER_COMPLETION` / `ROBOT_ADAPTATION` |
| DR | Table III | friction, mass, CoM, restitution, push after it 2000 | yes | none at first; Table III scaled to Mini-Pi afterwards, one family at a time | per user instructions |
| Iterations | 8000 | 1500 | yes | 8000 | paper |
| Timing | 200 Hz sim / 50 Hz policy / 500 Hz hardware | same | no | **500 Hz sim** / 50 Hz policy | `HARDWARE_CONSTRAINT` / `SIMULATOR_PORT` (Mini-Pi chatters at 5 ms). The Mini-Pi hardware runs the firmware PD with targets refreshed at 1 kHz; 500 Hz simulated PD is an **approximation requiring sensitivity validation** (e.g. 1 ms physics check), not exact equivalence |

## 9. Force-guided objective (Eq. 5–8): what is fixed, what is ambiguous

Fixed by the paper:

- Two constraint costs `C1 = F`, `C2 = T` (Eq. 4 quantities) with `J_Ci(π) ≤ d_i`.
- Penalty-function form (Eq. 5) optimized with PPO; mixed advantage (Eq. 8):
  `Ā_t = A_t − Σ_i β_i ( J_{t,Ci}(π_k) + E[A_{t,Ci}]/(1 − γ) )`.
- `A` and `A_Ci` "are computed using GAE"; "prior to gradient calculation using the
  advantage functions, an additional standardization step is applied" (the advantage
  functions, not `J`).
- Baseline penalty factor **β = 0.001** (Table IV(b): 0.02 degrades).
- `d_i = 0` at `t = t_tag`, so that `Ā = A` after the assist ends.

`PAPER_AMBIGUITY` (to be derived before coding, with a hand-computed unit test):

1. **Estimator of `A_Ci`.** GAE needs a value baseline for each cost; the paper names
   none. Candidates: (a) a separate cost value estimator; (b) GAE with a zero baseline
   (truncated discounted cost sum); (c) another estimator. The paper does not require a
   separate cost critic; if one is introduced it is an implementation completion and
   must be justified, isolated from the reward critic and tested.
2. **Meaning and estimator of `J_{t,Ci}(π_k)`.** The text calls it "the constraint
   function value at time step t in the trajectory". It could be the immediate cost, a
   per-step return estimate, or a value estimate. No choice is made yet; equating it
   with a cost value estimate `V_Ci(s_t)` is not supported by evidence.
3. **Standardization scope.** Which quantities are standardized (only `A`; `A` and
   each `A_Ci`), and whether before or after the β-weighted combination. `J` is not
   assumed standardized.
4. **Cost units.** Using normalized costs `‖F‖/F_max` and `‖T‖/(T_max·π)` is an
   implementation hypothesis, not a paper fact. β = 0.001 is preserved as the paper
   baseline, but its effective strength depends on the cost units and on the
   standardization scope: any unit change changes the meaning of β and must be stated.
5. **Teacher vs student terms** (`J^j`, `A^j` per group in Eq. 8): whether costs and
   advantages are computed per group or jointly.

The release provides no evidence for any of these (§6: its force path is inactive and
has a different, β-free form).

## 10. Mini-Pi numbers derived from the reference (pending the plant)

| Quantity | Reference | Scaling | Mini-Pi |
|---|---|---|---|
| Mass | 27.669 kg | measured | 6.94 kg (68.1 N) |
| Stance height L | 0.75 m | measured | 0.345 m |
| F_max | 400 N = 1.474 mg | ×(m g) | 100.4 N |
| T scale | release world-x coefficient 10 N·m (not `T_max`) → 10/(271.4·0.75) = 0.0491 | ×(m g L) | order-of-magnitude anchor 1.15 N·m only; `T_max` **UNRESOLVED** |
| μ | release has none (linear) | – | **UNRESOLVED** |
| Stage thresholds | g2 below 0.35 (0.467 L), g3 above 0.40 (0.533 L); g4 needs `force_scale < 100 N`, which depends on height **and** `time_coeff` (no fixed height) | ×L | g3 threshold ↔ 0.184 m; no h2 derivable from the release |
| Height targets | 0.3 / 0.45 / 0.6 / 0.75 | ×L/0.75 | 0.138 / 0.207 / 0.276 / 0.345 |
| Mini-Pi landmarks | – | measured (old audit) | lying 0.085, lowest balanced crouch 0.19, stance 0.345 |
| Base-height σ | 0.12 m | ×L | 0.055 m |
| Lin-vel tracking gate | h > 0.6 (0.8 L) | ×L | 0.276 m |
| Wheel/foot force kernel | 0.05/N at 10·M = 276.7 N (20 N = 0.072 Mg) | ×(M g) | 1/(0.072·68.1 N) |
| Feet distance band | 0.32–0.38 around 0.35 | ×d0 | 0.914–1.086 × Mini-Pi nominal foot spacing |
| Power gate | h < 0.7 (0.933 L) | ×L | 0.322 m |
| Commands | vx [−0.5, 0.8], yaw ±0.3, vy 0 | Froude √(L/0.75) = 0.678 | vx [−0.34, 0.54], yaw ±0.44 |
| Reset height | 0.1 m | – | measured settled poses |
| Unactuated window | 30 policy steps = 0.6 s (actions 0 = hold default pose; assist and obs masked) | same | 30 steps; check the settle on Mini-Pi |

Stage heights h1, h2, h3 remain **candidate Mini-Pi parameters**. Inputs for the
mapping: the release height progression (targets 0.3/0.45/0.6/0.75, g3 switch at 0.40,
i.e. 0.138/0.207/0.276/0.345 and 0.184 m after stance scaling) and Mini-Pi landmarks
(lying 0.085, lowest balanced crouch 0.19, stance 0.345, to be re-measured on the final
plant). The release gives no fixed second threshold (g4 is force/time based), so h2
needs an independent, documented argument before it is fixed.

Dimensioned reward coefficients (`torques`, `dof_acc`, `power`, `action_rate`, …) are
**not** rescaled: the old `TORQUE_SCALE = 36` is dropped.
