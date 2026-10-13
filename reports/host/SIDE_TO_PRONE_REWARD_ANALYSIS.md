# Side-fall recovery: reward and training analysis

Analysis date: 2026-10-13. Checkpoints: original rev13 `model_1100` and resumed `model_2700` in `20261013_162942_rev13_floor025_from1800`. This report examines why a side-fallen robot might roll toward prone before standing. Training code and reward weights were not changed.

The objective rewards successful foot-supported standing, not preservation of the original fall orientation. Rolling to a familiar, more symmetric recovery pose is therefore an allowed solution. Several rewards make a direct asymmetric side rise expensive, while low torso contact and slow rotation remain relatively cheap. **There is no explicit prone target, no side-to-prone state machine, and no requirement to recover directly from the side.**

## Measured routes and scope

**The measured behavior is broader than side → prone → stand.** In model_2700, left-side starts much more often pass through supine-like orientation. The route is not fixed, and the left/right strategies differ. All 300 primary episodes per checkpoint reached first standing; this does not establish sustained stability.

| Checkpoint / initial side | Episodes | Sustained prone visit | Sustained supine visit | Already prone at actuation | Mean stand time |
|---|---:|---:|---:|---:|---:|
| 1100 / left | 59 | 3.4% | 30.5% | 5.1% | 1.308 s |
| 1100 / right | 53 | 22.6% | 9.4% | 1.9% | 1.275 s |
| 2700 / left | 59 | 23.7% | 96.6% | 5.1% | 1.598 s |
| 2700 / right | 53 | 9.4% | 9.4% | 1.9% | 1.679 s |

These percentages are orientation visits, not mutually exclusive route labels. The detector also counts a prone-like initial actuated pose; the “already prone” column quantifies that distinction. Some brief or partial transitions do not pass the detector. Not passing through prone or supine does not by itself prove a direct side rise.

**Matched action-scale interventions (first 100 episodes, same seed):**

| Policy | Fixed action scale | Mean stand time | Mean slip distance per sole before standing | Left prone / supine counts (13 episodes) | Right prone / supine counts (17 episodes) |
|---|---:|---:|---:|---:|---:|
| 1100 | 0.466 | 1.207 s | 23.41 cm | 1 / 2 | 3 / 2 |
| 1100 | 0.250 | 1.683 s | 25.93 cm | 1 / 2 | 5 / 0 |
| 2700 | 0.250 | 1.588 s | 27.95 cm | 4 / 12 | 1 / 1 |
| 2700 | 0.466 | 1.351 s | 25.49 cm | 3 / 10 | 2 / 1 |

Reducing 1100’s scale from 0.466 to 0.25 increased mean stand time by about 39% and mean sole travel by about 11% on these matched 100 episodes. Increasing 2700’s scale shortened mean stand time by about 15% and reduced mean sole travel by about 9%, while its left-side supine visits remained common (12/13 → 10/13). Thus scale influences timing and sliding, but does not explain the whole learned route. Higher scale also increased measured overspeed and rotation costs; these results are not a recommendation to deploy it.

**Reward paid/earned during side recovery, mean accumulated contribution until first standing:**

| Term | 1100 left | 1100 right | 2700 left | 2700 right |
|---|---:|---:|---:|---:|
| task_product | 16.46 | 16.03 | 22.44 | 28.03 |
| style_ground_parallel | 6.77 | 8.00 | 9.28 | 13.47 |
| style_one_foot_rise | -1.65 | -1.68 | -2.77 | -1.83 |
| style_recovery_rotation | -5.70 | -4.17 | -11.48 | -7.19 |
| style_nonfoot_support | -0.72 | -0.70 | -0.84 | -0.67 |
| style_knee_load_support | -3.75 | -2.43 | -4.98 | -2.29 |
| style_loaded_foot_translation | -5.36 | -4.29 | -6.11 | -6.10 |
| style_loaded_sole_slip | -1.59 | -1.50 | -2.70 | -3.13 |
| style_joint_overspeed | -4.18 | -4.53 | -3.64 | -3.02 |

The task row is the actual multiplicative task product summed over time, before PPO group weighting/normalization. Other rows already include dt and their configured coefficient. `style_ground_parallel` is a sizeable positive shaping contribution, while nonfoot contact is relatively inexpensive in the observed low phase. Model_2700 still pays substantial rotation and translation costs; it is not avoiding all penalties. A longer trajectory has more steps to accumulate both positive and negative reward, so these totals alone cannot rank route quality.

**Strictly frozen primary slip comparison:** 1100 averaged 19.80 cm left / 26.50 cm right; 2700 averaged 26.03 cm left / 28.47 cm right. Their mean time to first standing was 1.236 s versus 1.584 s. These replace the earlier approximate comparison for this audit.

Example orientation traces: [side_recovery_orientation_traces.png](side_recovery_orientation_traces.png). These are the first recorded non-prone-visit example for each side/checkpoint, not population averages or median episodes.

The route detector uses the repository's canonical prone convention: projected gravity in the body frame has `g_x = +1` in prone, `g_x = -1` in supine, and `g_z = -1` when upright. A prone visit requires `g_x >= 0.8` continuously for at least 0.10 s before first standing. This captures a substantial prone-like orientation, within about 37 degrees of canonical prone; it does not prove belly contact or detect every partial forward lean. A supine visit uses `g_x <= -0.8` with the same duration. Both can occur in one episode.

Standing means base height > 0.30 m and upright > 0.8. Times begin at actuation. The primary comparison uses 300 episodes per checkpoint, matching reset seeds (20261013–20261015), the same asset and current reward implementation, no pull force, domain randomization, additive observation noise, stance starts, or drag/velocity termination. Each checkpoint uses its saved mean action scale and saved reward-success EMA. Reset callbacks are explicitly frozen. This diagnoses deployed behavior; it does not reproduce the original 1100 training environment or isolate individual reward causes.

The two additional scale experiments use 100 episodes each with seed 20261013. They are controlled diagnostics, not training runs or statistical guarantees. The environment still adds ±0.025 jitter to the action-scale observation even when additive observation noise is disabled.

## Reward mechanisms that can favor the route

Raw coefficients below are configuration values. Additive constraints are multiplied by policy dt = 0.02 s. Multiplicative task factors are **not** multiplied by dt. PPO normalizes each critic's advantage separately and combines them with weights task 2.5, regularization 0.1, style 1, target 1. Raw coefficients or accumulated group returns therefore cannot be compared as a single scalar objective without accounting for normalization.

| Mechanism | Relevant terms / coefficient | How it can favor rolling before rising |
|---|---|---|
| Both feet must carry load | `style_one_foot_rise` −20; `target_two_foot_flat_support` +6; foot load +8 | A side rise often begins on the lower leg while the upper leg is unloaded. As the torso rises through ~0.10 m, one-foot support is penalized. Rolling can make it easier to bring both legs into a shared sagittal recovery. This geometric advantage is a hypothesis, not a measured counterfactual. |
| Height credit depends on support | task head height; target base height +10; supported height +8 | `credit = 0.2 + 0.3*support_gate + 0.5*ready`, with `ready = sqrt(coverage_pair*com_score)*support_gate`. Unsupported/one-foot recovery gets only the floor; two-foot support already improves credit before complete readiness. Moving into a pose that makes both soles usable can beat a direct side lift. |
| Whole-body COM over feet | COM +6, COM margin +2, torso centering +4, feet under COM +3 | Lifting a sideways torso while keeping its COM inside a narrow support polygon is difficult. Rolling low can reposition the torso and feet before earning height. These terms reward geometry, not the route taken. |
| Flat, planted soles | coverage +8, deficit −10, tilt −8 | A loaded sole resting on an edge pays a deficit/tilt cost. From a side pose, the lower foot may be tilted and the upper foot elevated. A symmetric pitch-plane recovery can make sole placement easier. The costs use settled load, so fresh touchdown is charged gradually. |
| Inherited ankle geometry reward | `style_ground_parallel` **+20** | This relatively large style reward tests vertical variance of five ankle-roll/auxiliary frame origins within each leg (`var < 0.05` after z is multiplied by 10). It is not a terrain-contact test and is active during low recovery. It can reward configurations with each ankle assembly parallel to the ground even before load-bearing. It does not explicitly compare left/right foot heights or target prone. Its geometric alignment could favor rolling out of a side orientation. The auxiliary frames include offsets at ±10 cm laterally, wider than the actual sole (~±3 cm), so this is an inherited proxy for foot orientation rather than sole-contact geometry. |
| Low body contact is allowed | `style_nonfoot_support` −15 | Its gate is `sigmoid((h−0.20)/0.05) * (0.2 + 0.8*sigmoid((upright−0.35)/0.15))`. At h=0.10 m and upright=0 the gate is only about 0.032. Thus normalized nonfoot support of 1 costs about −0.0096 per step, versus −0.30 at a fully open gate. This makes early torso-supported rolling affordable; the exact contact and gate vary per episode. |
| Rotation is limited by speed, not route | `style_recovery_rotation` −2 | Cost is squared angular-speed excess above 1.5 rad/s, times `(0.25 + 0.75*curriculum_strength)` and a recovery/standing gate. Slow rolling below the threshold is free for this term. It penalizes fast angular motion, not total angle rotated, prone visits, or direction reversals. |
| Fast direct ascent is costly | ballistic ascent −80, unsupported ascent −6, rise speed −4 | These discourage throwing the torso up before support is established. COM rise speed above 0.20 m/s pays the rise-speed cost; unsupported ascent begins above 0.05 m/s. A gradual low reorientation can be cheaper than a dynamic side push. |
| Brief knees are allowed | knee support −45 with ramp 0.25→1 over 0.8 s | A prone route can briefly use knees without paying the full sustained-support penalty. Low stationary contact is also softened by the inherited height/lifting gate. Prolonged kneeling is discouraged, so this is permission for a transient, not an explicit knee reward. |
| Crossing and large stance are costly | leg contact −6, stance width −4, feet distance −10 | An asymmetric side rise may cross legs or separate feet excessively. Stance width is reduced to 25% while rising, permitting placement followed by adjustment. Leg contact is lift-gated, so low repositioning is cheaper. |
| Standing terms do not select the initial route | posture −8, knee extension −16, load imbalance −6, stable standing −1, anchor −4, stance return +4, step placement +100 | Most of their influence opens with the standing gate or applies at standing touchdown. They select the resulting stance and stability, not direct versus rolled side recovery. The +100 step impulse still receives dt scaling. The anchor follows feet before full standing, so it is not a fixed pre-rise no-drag constraint. |

The strongest structural explanation is the combination of **costly asymmetric rising + affordable low reorientation + dense credit for reaching a shared two-foot standing solution**. The ground-parallel term is an additional candidate to ablate because it rewards ankle geometry without requiring actual sole load.

The task orientation reward depends on `−g_z`, so canonical side, prone and supine poses all have approximately the same value at 90-degree tilt. It cannot by itself select prone over supine. The head-height factor measures head height relative to mean foot height, so head/foot placement and the robot's fore-aft geometry can introduce a preference once the limbs move. This distinction matters: an uprightness objective is not a prone preference.

## Other contributors outside the reward

### Reset distribution and shared policy

The fallen-reset generator samples five classes uniformly: supine, prone, left, right, intermediate. With 15% stance starts, the expected mixture is roughly 17% for each fallen class and 15% standing starts. Prone is directly practiced, and intermediate orientations expose the same actor to states near it. A single feedforward policy serves every class; no fall-class label or requested recovery direction is supplied. Reusing a familiar recovery route can be easier to learn than maintaining separate routes for left/right falls. This is a plausible learning explanation, not proof that a particular training sample created the behavior.

Initial joints are randomized within fixed baseline-derived spans. Root orientations include random yaw; the robot starts at 0.34–0.37 m with small linear/angular velocities. The first **0.60 s** (30 policy steps) are unactuated, allowing the robot to settle under gravity and zero-action damping before the policy receives nonzero observations. An initially side-labeled episode can already lean toward prone or supine at actuation. Route measurement therefore records gravity at the first actuated step instead of assuming the reset class describes that pose exactly.

### Observations and partial information

Each 43-value observation contains body angular velocity, projected gravity, 12 joint positions, 12 joint velocities, previous actions, and action scale. Six frames produce 258 inputs, roughly 0.12 s of sampled history. There is no explicit foot force, sole coverage, COM, support polygon, body height, base linear velocity, route command, or fall-class observation. The actor can infer some contact state from motion but cannot directly see the quantities used to shape rewards. A familiar repeatable transition can be easier to control under this partial information than an unfamiliar asymmetric support transfer.

Yaw is not represented as absolute heading through projected gravity. Random yaw makes recovery approximately heading-independent; “turning prone” means changing body tilt, not necessarily choosing a world heading. Playback must be interpreted in body coordinates.

### Actuation and dynamics

Actions are incremental joint commands: `q_target = q_current + action*action_scale`, followed by PD torque. The action clipping limit is 100, so action scale alone is not an absolute joint-motion bound and the actor can partly compensate with larger outputs. The controller runs at 50 Hz; simulation/torque monitoring runs at 200 Hz.

PD gains vary substantially by joint: KP hip pitch 60, hip roll 40, thigh 20, calf 60, ankle pitch 30, ankle roll 10; KD 2.4, 0.8, 0.4, 2.8, 1.6, 0.3 respectively. Effort limits are 16 Nm except ankle roll at 6 Nm. Accelerating torque fades from full at 3 rad/s to zero at 4 rad/s; braking remains available. Joint coupling and impacts can exceed this speed, so it is not a hard velocity cap. Limited ankle-roll authority and motor-speed shaping could make a direct lateral balance/lift harder than a slower pitch-plane route; this needs a controlled mechanics comparison.

The saved action scale is ~0.466 for 1100 and 0.25 for 2700. Exploration also shrank: the 2700 checkpoint generally has smaller learned action standard deviations. Since physical perturbations scale with both policy std and action scale, the resumed run may explore alternative routes less readily. PPO's entropy coefficient is 0.01; gamma is 0.99 per 20 ms step and lambda is 0.95. Discounting gives earlier standing rewards an advantage, but there is no explicit time cost, route-length cost, or terminal bonus for direct side recovery. Constant per-step positive shaping can also reward a useful intermediate posture while the policy spends time there.

### Asset and collision representation

Both checkpoints record the same asset SHA256 `ea7d909d9a0516d18f84d43a9bbbc5a1d03b109e11e15e1793dd527fe439ebb0`. The XML describes a legged robot without actuated arms to push against the ground. Hip-roll outward ROM extends to 120 degrees, knees range −2.5…1.65 rad, and inward thigh yaw is constrained. These permit unusual folding and wide leg placements; they do not guarantee direct side recovery.

The torso's terrain collision is a box, and sole contact is modeled with five capsules per foot. Thigh terrain collision is effectively disabled by a 0.5 mm box; calf terrain collision is a local knee housing rather than the full shin. Added thigh/shin/foot/torso shells protect leg-leg, leg-torso and own-foot/hip interactions, **not full mesh-versus-ground contact**. That means ground leverage and support during low rolling can differ from the visible meshes. Visual side/prone appearance alone is insufficient to infer which bodies actually bear weight. Contact simplification is a material candidate, not evidence that the policy is exploiting a specific collision error.

Training randomizes friction, restitution, mass/COM, motor strength, PD gains, torque offsets and delay. A controller may favor routes that tolerate these variations. Actual sliding friction is `0.5*(robot_friction + 0.75)`: training robot friction 0.1–1.0 gives effective 0.425–0.875, while non-randomized evaluation uses effective 0.65. These affect which low-contact/support-transfer motions are feasible; no friction ablation was performed. The configured push option is not actually applied by the inherited release code.

Small left/right mass, inertia and joint-range differences exist, and the learned network has no enforced mirror symmetry. Initial random poses and learned strategy can therefore produce unequal left/right route frequencies.

### Curricula, training lineage and playback

Early training uses a vertical pull force on the base only once upright (`g_z < −0.8`), with 40→0 N progression. It does not explicitly pull into prone or teach how to leave a side pose, but it can make reaching the upright gate disproportionately useful early in learning. Both inspected checkpoints have zero saved force, and all report rollouts disable assistance.

The original 1100 checkpoint predates the collapse associated with post-standing drag termination. The resumed lineage turns drag termination off and introduces an action-scale floor; the current saved run/code uses **0.25**, even though historical reward notes still say 0.30. A source diff between the recorded checkpoint commits shows those behavioral changes, not a redesign of the reward coefficients. Continuing training from 1100 preserves the network, optimizer, and curriculum state, so existing recovery strategies are inherited rather than relearned from scratch.

Success-dependent shaping strengthens rotation, regularization and standing penalties. At EMA 0.60 the curriculum begins; it reaches full strength at 0.85. The saved EMA is about 0.501 at 1100 and 0.751 at 2700. This changes the optimization environment even with unchanged raw coefficients. It does not selectively reward prone. Velocity termination remains active in training (12→5 rad/s as success increases), while playback disables it; slower routes may be less likely to trigger it. Disabled drag termination concerns **post-standing** drag and should not be described as a direct cause of pre-standing prone rolling.

## Measurement limitation found during this audit

The inherited `reset_idx` calls `update_force_curriculum` and updates the success EMA even when `env.test` is true. Existing evaluation helpers set `env.test`, but do not explicitly freeze those callbacks. With multiple trials, manual resets and automatic resets can therefore change action scale and reward strength. “Fixed action scale” in their description is not completely enforced. This explains why earlier slip comparisons should be treated as approximate operational measurements rather than strictly frozen controls.

The new diagnostic freezes both callbacks, restores the checkpoint's success EMA, and logs the multiplicative task product separately. Task factor sums (`task_orientation`, `task_head_height`) are **not** additive task returns. Measured reward contributions compare what the chosen trajectory paid/earned; they cannot establish whether an untried direct route would have achieved a better return.

## Recommendations and discriminating experiments

1. Decide whether rolling is acceptable. If the goal is reliable standing with low slip, prone passage is not automatically a failure. Compare time, foot/body travel, impact, velocity, knee use and final stability across routes before changing rewards.
2. If direct side recovery is required, evaluate a paired ablation of `style_ground_parallel` (+20) and the one-foot-rise penalty during early side recovery. Keep final two-foot standing requirements. Measure whether route frequency changes without increasing falls or slip. Simply adding a large no-roll penalty can remove a viable recovery path.
3. Provide foot-contact/load observations if the intended behavior requires accurate unilateral support transfer. This changes the actor input dimension and requires retraining or migration; it is not a checkpoint-compatible playback tweak.
4. Use matched side-reset poses and frozen scale/EMA for comparisons; distinguish passive settling, partial forward lean, sustained prone orientation, and belly contact. Freeze curricula explicitly in evaluation helpers before using them for definitive policy rankings.
5. For less slipping, assess pre-rise loaded sole travel separately from post-standing anchor drift. Raising a standing-only anchor cost will not directly constrain the entire pre-rise route. Explore load-aware stepping/placement credit and sufficient exploration rather than using early termination as the first fix.
6. Check full ground collision geometry and achievable lateral support/torque in a mechanics study before demanding direct side recovery. The current no-arm asset and simplified thigh/shin ground contact may make that route materially harder.

No individual reward ablation or alternate direct-side controller was trained in this analysis. The ranked mechanisms are code-supported hypotheses; route frequencies and scale interventions are measured evidence. No production reward, observation, asset, or training configuration was modified.

## Reproduction and artifacts

Diagnostic: [tools/analyze_side_recovery.py](../../tools/analyze_side_recovery.py).

```bash
source tools/activate_host_env.sh
python tools/analyze_side_recovery.py \
  --checkpoint logs/host_staged_recovery/20261013_162942_rev13_floor025_from1800/model_2700.pt \
  --out reports/host/rev13_model2700_side_recovery.json
```

Evidence: [1100 routes/rewards](rev13_model1100_side_recovery.json), [2700 routes/rewards](rev13_model2700_side_recovery.json), [1100 at scale 0.25](rev13_model1100_scale025_side_recovery.json), [2700 at scale 0.466](rev13_model2700_scale0466_side_recovery.json).

Primary source files: [reward composition, observations, original terms](../../src/minipi_getup/host/env.py), [inherited configuration](../../src/minipi_getup/host/config.py), [safe-recovery gates/reset distribution](../../src/minipi_getup/host_safe_recovery/env.py), [safe-recovery coefficients](../../src/minipi_getup/host_safe_recovery/config.py), [staged terms and callbacks](../../src/minipi_getup/host_staged_recovery/env.py), [staged coefficients](../../src/minipi_getup/host_staged_recovery/config.py), [multi-critic advantage normalization](../../src/minipi_getup/host_rl/rollout_storage.py), [asset XML](../../src/minipi_getup/asset_zoo/robots/hightorque_minipi/xmls/cl_pai_host_torso_host_legs_hip120_legcoll.xml), [resume implementation](../../src/minipi_getup/host_staged_recovery/train.py).

## Complete additive coefficient inventory

From the model_2700 run’s saved `reward_scales_raw`. Zero terms are disabled. Multiplicative task factors are orientation × head height (unit scales), separately documented above.

| Additive term | Raw coefficient | dt-scaled coefficient |
|---|---:|---:|
| `regu_action_rate` | -0.03 | -0.0006 |
| `regu_dof_acc` | -2.5e-07 | -5e-09 |
| `regu_dof_pos_limits` | -100 | -2 |
| `regu_dof_vel` | -0.001 | -2e-05 |
| `regu_dof_vel_limits` | -1 | -0.02 |
| `regu_joint_overspeed` | 0 | 0 |
| `regu_joint_power` | -2.5e-05 | -5e-07 |
| `regu_joint_tracking_error` | -0.00025 | -5e-06 |
| `regu_smoothness` | -0.03 | -0.0006 |
| `regu_torques` | -2.5e-06 | -5e-08 |
| `style_ballistic_ascent` | -80 | -1.6 |
| `style_feet_distance` | -10 | -0.2 |
| `style_ground_parallel` | 20 | 0.4 |
| `style_hip_roll_deviation` | 0 | 0 |
| `style_hip_yaw_deviation` | -10 | -0.2 |
| `style_joint_overspeed` | -2 | -0.04 |
| `style_knee_deviation` | -0.25 | -0.005 |
| `style_knee_load_support` | -45 | -0.9 |
| `style_left_foot_displacement` | 2.5 | 0.05 |
| `style_leg_contact` | -6 | -0.12 |
| `style_loaded_foot_translation` | -4 | -0.08 |
| `style_loaded_hip_roll_excess` | 0 | 0 |
| `style_loaded_knee_hyperextension` | 0 | 0 |
| `style_loaded_sole_coverage_deficit` | -10 | -0.2 |
| `style_loaded_sole_slip` | -6 | -0.12 |
| `style_loaded_sole_tilt` | -8 | -0.16 |
| `style_nonfoot_support` | -15 | -0.3 |
| `style_one_foot_rise` | -20 | -0.4 |
| `style_recovery_rotation` | -2 | -0.04 |
| `style_right_foot_displacement` | 2.5 | 0.05 |
| `style_rise_speed` | -4 | -0.08 |
| `style_stance_width` | -4 | -0.08 |
| `style_standing_foot_anchor` | -4 | -0.08 |
| `style_standing_knee_extension` | -16 | -0.32 |
| `style_standing_load_imbalance` | -6 | -0.12 |
| `style_style_ang_vel_xy` | 1 | 0.02 |
| `style_unsupported_ascent` | -6 | -0.12 |
| `target_ang_vel_xy` | 10 | 0.2 |
| `target_com_support` | 6 | 0.12 |
| `target_com_support_margin` | 2 | 0.04 |
| `target_default_posture` | -8 | -0.16 |
| `target_feet_height_var` | 2.5 | 0.05 |
| `target_feet_under_com` | 3 | 0.06 |
| `target_foot_load_support` | 8 | 0.16 |
| `target_foot_supported_height` | 8 | 0.16 |
| `target_knee_sole_progress` | 0 | 0 |
| `target_lin_vel_xy` | 10 | 0.2 |
| `target_loaded_knee_flexion` | 0 | 0 |
| `target_recovery_hip_roll` | 0 | 0 |
| `target_sole_contact_coverage` | 8 | 0.16 |
| `target_stable_standing` | -1 | -0.02 |
| `target_stance_return` | 4 | 0.08 |
| `target_step_placement` | 100 | 2 |
| `target_target_base_height` | 10 | 0.2 |
| `target_target_orientation` | 10 | 0.2 |
| `target_torso_center_over_feet` | 4 | 0.08 |
| `target_two_foot_flat_support` | 6 | 0.12 |
