# host_staged_recovery — reward definitions

Variant of `host_safe_recovery` (HoST multi-critic PPO, unchanged) shaped toward the recovery sequence

> legs in → feet down → COM over the support → extend and raise the torso → default posture

as a soft preference, not a state machine. No reward gate is a hard switch, there is no time penalty, and
no term targets a fixed knee or hip angle.

## Conventions

- HoST multiplies every constraint scale by the policy `dt = 0.02 s`. Groups are `task` (multiplicative),
  `regu`, `style` and `target`, with group weights `[2.5, 0.1, 1, 1]`. Each group has its own critic and
  advantages are normalized per group.
- `W`: randomized robot weight (N). `L`, `R`: per-foot terrain normal load, 20 ms mean.
  `sup(F) = 1 − exp(−relu(F − 0.02W) / 0.25W)` is the normalized support in [0, 1).
- **COM**: the whole-body centre of mass, `data.subtree_com[root]`, in world frame. `v_z` is its finite
  difference over one policy step.
- **Sole footprint**: in the `ankle_roll_link` frame, the bottom lines of the 5 sole capsules span
  x ∈ [−0.129, 0.026] m and half-width 0.029 m (toe) / 0.0233 m (heel), at z = −0.048 m.
  - The 4 corners are transformed with the real foot pose.
  - A corner counts as support if it lies within 1.2 cm of the ground **and** its foot has `sup > 0.15`.
- **Support polygon**: the convex hull of the counted corners. It covers one foot, two feet, or a line
  for an edge contact.
  - `d_out`: distance from COM_xy to the hull (0 inside). `d_c`: distance from COM_xy to the centroid of
    the counted corners.
- `com_score = exp(−(d_out/0.03)²) · (0.7 + 0.3·exp(−(d_c/0.06)²))`, or 0 without support.
- **Stance**:
  - `width` is the horizontal distance between the sole centres (0.16 m at zero joints).
  - `gap` is the lateral clearance between the two footprints, measured along the torso's horizontal
    left axis (falling back to the mean foot heading when the torso is rolled past ~73°). It is 0.10 m
    at zero joints; ≤ 0 means touching or crossing.
- **Gates**:
  - `lift = σ((z_base − 0.14)/0.03)`: torso off the floor.
  - `standing gate = σ((z_base − 0.25)/0.045)·σ((−g_z − 0.65)/0.12)`, the same as `host_safe_recovery`.

## Revision 2 (after run `20261012_145629`)

At iteration 700, the first run stood on one foot: 94% of the load on the right foot, with the left foot
1.3 cm above the ground and resting on the right one. Every per-foot cost was weighted by that foot's own
load, so an unloaded foot escaped all of them. Revision 2:

- regates `style_stance_width` on `max(sup)·lift`;
- regates the knee-extension cost (then named `style_knee_reverse_lock`) on `max(sup, standing gate)`;
- adds `style_standing_load_imbalance`.

Replaying those standing poses on the real meshes showed the two feet interpenetrating (median 5 mm,
up to 21 mm). The baseline collision model is smaller than the meshes: feet ±2.5–3.1 cm vs a ±3.6 cm
mesh, and the shin and thigh have no collision at all. Revision 2 therefore also:

- moves to `cl_pai_host_torso_host_legs_hip120_legcoll.xml`, which adds massless box shells on the
  mesh AABBs of the thigh, shin and foot. They use bits l=2/4 and r=4/2, so they collide left-vs-right
  only (no terrain, torso or same-leg contact). Mass and inertia are unchanged;
- adds `style_leg_contact` from a left-subtree vs right-subtree contact sensor;
- restricts it to the standing gate, raises it to −16, and renames it `style_standing_knee_extension`;
- adds the training-only velocity termination (off, then 20 → 6 rad/s with the success curriculum);
- measures `gap` across the torso's lateral axis instead of the mean foot heading.

## Removed relative to `host_safe_recovery`

| Term | Was | Reason |
|---|---|---|
| `target_loaded_knee_flexion` | +4 | joint-specific strategy reward |
| `target_knee_sole_progress` | +6 | knee-flexion reward |
| `target_recovery_hip_roll` | 0 | hip-roll opening reward |
| `style_loaded_hip_roll_excess` | −12 | joint-specific; the hip ROM was widened on purpose |
| `style_loaded_knee_hyperextension` | −4 | replaced by `style_standing_knee_extension` |
| `regu_joint_overspeed` | −0.30 | moved to `style_joint_overspeed` |
| `style_hip_roll_deviation` (HoST) | −10 | HoST limit of 1.4 rad (one leg) / 0.9 rad (both) would cap the widened hip-roll ROM |

## Pipeline terms (new or modified)

| Stage | Term | Scale | Value per step | Active when |
|---|---|---:|---|---|
| 1 legs in | `target_feet_under_com` | +3 | mean over feet of `exp(−(relu(‖foot_xy − COM_xy‖ − 0.12)/0.08)²)` | always (also satisfied when standing) |
| 2 feet down | `target_two_foot_flat_support` | +4 | `share₂ · max(sup_L, sup_R) · flat_L · flat_R`, with `share₂ = clip(min(F_L,F_R)/(F_L+F_R) / 0.3)` and `flat = exp(−(tilt/0.15)²)` | always; one-foot standing (4% share) earns ~0.13 |
| 2 feet down | `style_stance_width` | −4 | `min(4, (relu(0.04 − gap)/0.03)² + (relu(width − 0.26)/0.08)²) · max(sup_L, sup_R) · lift` | either foot loaded, torso up; a foot resting on or crossing the other also pays |
| 3 COM transfer | `target_com_support` | +6 | `com_score · share₂ · lift` | both feet loaded, torso up |
| 4 rise | height credit | × | head height (task), `target_base_height`, `foot_supported_height` × `0.05 + 0.95·ready`, `ready = feet_under_com · share₂ · min(flat) · com_score · exp(−(d_centre/0.03)²)` | always; rising pays fully only after feet are under the COM, both feet are down and the COM is supported; floor 0.05 keeps an early gradient |
| 4 rise | `style_one_foot_rise` | −20 | `max(sup_L, sup_R) · (1 − share₂) · rise_gate`, `rise_gate = σ((h − 0.10)/0.02) · (1 − standing gate)` | actuated; knee/body support with no foot load is not charged here |
| 4 rise | `style_unsupported_ascent` | −6 | `(relu(v_z − 0.05)/0.20)² · (1 − ready) · lift` | after actuation starts |
| 5 standing | `style_standing_knee_extension` | −16 | mean over legs of `standing gate · (relu(−q_calf − 0.25)/0.40)²`: 0 above ~22° flexion, 1.0 at the straight leg, 31.6 at the −2.5 reverse limit | standing only: straight or reversed knees are free while getting up but must return toward the flexed default once standing (static standing at calf 0 needs ~0.6 Nm at the knee) |
| 2 feet down / 5 | `style_leg_contact` | −6 | `sup_leg(‖F_left-leg↔right-leg‖) · lift`, with `sup_leg` using deadband 0.005W and scale 0.05W (~3.4 N) | after actuation, torso up |
| 5 standing | `style_standing_load_imbalance` | −6 | `((F_L − F_R)/(F_L + F_R + 0.1W))² · standing gate` | standing; ≈ 0.64 for a 64/4 N split |
| knee support | `style_knee_load_support` | −45 × ramp | the safe-recovery knee-housing load cost × `0.25 + 0.75·clip(t_knee/0.8 s)` | `t_knee` grows while knee support > 0.1 and decays with τ = 0.3 s otherwise |
| 4 rise | `style_rise_speed` | −4 | `min((relu(v_z,COM − 0.20)/0.10)², 9)` | actuated; a slow rise (~0.17 m/s) is free |
| velocity | `style_joint_overspeed` | −2.0 | `0.5·mean + 0.5·max` of `relu(peak₅ₘₛ|q̇| − 3)²` × regu ramp (1/3 → 1) | after the success curriculum |

**Knee geometry** (angle between the hip→knee and knee→ankle joint axes):

| calf (rad) | +1.65 | +1.00 | 0 (default) | −0.30 | −0.65 | −1.00 | −2.00 | −2.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| knee flexion | 132° | 95° | 37° | 20° | **0° (straight)** | −20° | −77° | −106° (reversed limit) |

The baseline/HoST limit of −0.65 is exactly the straight leg, so earlier "reverse knee" notes in this
project meant a straight knee locked on its stop. This asset widens the limit to −2.5 (106° backwards). It was briefly −2.95
(132°, symmetric with forward flexion), but a full-ROM sweep showed that with the ankle neutral the
foot hits the torso and hip motor from −2.59, so the limit stops just before that.

- Both directions are stopped by the hip-roll motor block at the top of the short (15 cm) thigh.
  Flexing forwards, only the slim shin reaches it, at about the +1.65 limit. Bending backwards, the
  foot (offset 7.5 cm forward of the ankle axis, ~17 cm long) swings into it from about −2.6 with the
  ankle neutral.
- Hip-motor shells (foot-vs-own-hip only) still stop the foot physically in combined poses: with the
  knee past ~−2.2 and the toe pointed down (ankle pitch ≈ −0.4) the box shells touch while the meshes
  are still ~2 cm apart (conservative, never a pass-through).
- With feet planted flat and the torso upright, the angle is symmetric: equal knee angles give equal
  torso heights in either direction.

**Hardware caveat:** in MuJoCo nothing blocks the reversed range, because adjacent bodies do not
collide and the convex hulls overlap at the knee anyway. Confirm that the real knee is not
mechanically stopped at the straight position before deploying a policy that uses it.

## Kept from `host_safe_recovery` and HoST (unchanged)

- **task:** `head_height × orientation`, HoST tolerance functions, × the height credit above.
- **target:**
  - `foot_load_support` (+8), `sole_contact_coverage` (+4), `foot_supported_height` (+8).
  - HoST post-task terms: `target_base_height`, `target_orientation`, `ang_vel_xy`, `lin_vel_xy`,
    `feet_height_var`.
  - `default_posture` (−8, standing gate, curriculum boost ≤ 2×), `stable_standing` (−0.05).
- **style:**
  - `loaded_sole_tilt` (−8), `loaded_sole_coverage_deficit` (−8, min coverage 0.9).
  - `loaded_sole_slip` (−6, linear above 0.03 m/s).
  - `nonfoot_support` (−15), `ballistic_ascent` (−80), `recovery_rotation` (−2).
  - HoST `ground_parallel`, `feet_distance` (> 0.45 m), `foot_displacement`, and
    `hip_yaw_deviation` / `knee_deviation`. These are the HoST originals (`hip_roll_deviation` is
    removed, see above).
- **regu:** all HoST regularizers, including `dof_pos_limits` (soft limit 0.9 of the XML range).

## Actuation, assets and curricula

- **Motor torque–speed limit:** the PD torque that accelerates a joint in its direction of motion fades
  linearly from the 16/6 Nm limit at 3 rad/s to 0 at 4 rad/s (revision 6; was 4.5 → 6). Braking torque is unaffected.
  - This roughly halves peak joint speed. It is **not** a hard cap: impacts and coupled motion can still
    exceed 6 rad/s, so evaluation reports the actuated peak against 6 rad/s.
- **Asset:** `cl_pai_host_torso_host_legs_hip120_legcoll.xml` (SHA256 `ea7d909d…`).
  - It is the baseline XML with hip roll widened outward to 2π/3 and the knee range widened to −2.5…1.65
    (reversible past straight), and thigh yaw toward the other leg capped at 0.28 (from 0.6); hip pitch stays −1.25…1.75 rad.
  - It adds the leg shells described in revision 2, a torso shell that copies the torso box, and
    hip-motor shells.
  - Bits: left 2/12 (left foot 66/12), right 4/10 (right foot 132/10), torso 8/6, left hip 16/64,
    right hip 32/128.
  - So the shells collide left-vs-right, leg-vs-torso and foot-vs-own-hip only.
  - A 1872-pose ROM sweep found no false contact at zero joints or fully adducted legs. The thigh shell
    never reaches the torso inside the ROM; feet and shins reach it only with the knee folded past
    ~0.8 rad and the hip flexed.
- **Reset:** the same distribution as the baseline. Joints are uniform within absolute half-spans equal
  to 25% of the baseline ROM, so the widened hip roll does not widen resets. Five classes: supine,
  prone, left, right, intermediate.
- **Velocity termination (training only):** an episode ends when the 5 ms peak actuated |q̇| exceeds a
  limit.
  - The limit is off until the success curriculum starts (EMA 0.60), then tightens from 12 to 5 rad/s (revision 6; was 20 → 6)
    with the same smoothstep (complete at EMA 0.85).
  - A smoke test showed that a fixed 12 rad/s limit ends every early episode: random policies peak at
    15–30 rad/s.
  - Evaluation and playback disable it.
- **Pull force:** HoST/safe curriculum unchanged: 40 N on `base_link`, +Z, once the torso is upright
  (`g_z < −0.8`), stepping 40 → 0 by 10 N on head-height progress. It is not gated on the COM.
- **Unchanged:** the action-scale curriculum (1.0 → 0.25 by HoST, then down to 0.10 on success) and the
  success-gated penalty curriculum.

## Stage monitors (logged, never used as reward gates)

Times are measured from the start of actuation:

- `t_feet`: first time both feet carry sustained support > 0.3.
- `t_com`: first time the COM is over an **adequate** support (both feet, `gap ≥ 0.04`,
  `width ≤ 0.26`, `d_out ≤ 1 cm`, `com_score > 0.5`) with the torso up.
- `t_rise`: the **last** upward crossing of base height 0.24 m with `−g_z > 0.5` before first standing.

Logged per episode:

- `stage_feet_before_rise`, `stage_com_before_rise` and `stage_supported_at_rise`. The last one says
  whether the support was adequate at the rise onset itself.
- `knee_contact_time_s`, `knee_extended_standing_frac`, `knee_reversed_standing_frac`, `com_support_error_m`, `final_stance_width_m`,
  `final_foot_gap_m`, `actuated_peak_joint_vel` and `actuated_velocity_violation` (> 6 rad/s).

`evaluate.py` computes the same events for any checkpoint, including the `host_safe_recovery` baseline,
with no pull force, DR or observation noise and a fixed action scale.

## Revision 4: strict pipeline order (feet under COM → both feet down → rise)

The revision-3 run (`20261012_173122`, 600 iterations) still stood on one foot: the weaker foot carried
4% of the load, the legs leaned on each other for 6 s per episode, and the two-foot success EMA stayed at
0, so the penalty curriculum and velocity termination never started. Cause: every rising credit used
`com_score`, which is satisfied by the hull of a single loaded foot, and the two-foot reward used
saturating supports that are already near 1 at a few percent of the weight.

- `ready = feet_under_com · share₂ · min(flat_L, flat_R) · com_score` (product: every step is needed).
- Height credit `0.15 + 0.85·ready` (floor 0.4 → 0.15); unsupported ascent uses `1 − ready`.
- `target_com_support` and `target_two_foot_flat_support` use the load share `share₂`.
- Monitor `stage_ready_before_stand`: fraction of stood episodes with `ready > 0.5` before first standing.

## Revision 5: getting up must use both feet in every orientation

`model_900` of the rev-4 run stood on both feet (weaker foot 42–44%) but rose on one: between base height
0.10 m and standing, both feet carried ≥10% of the weight in only 18–36% of the steps, one foot in 33–47%
(mostly the left, the other foot ~5–10 cm up), in all five orientations. From supine it sat up, splayed the
left leg out (hip roll ~1.6) and pistol-squatted while the right knee reversed with the foot in the air;
the right foot landed only at ~0.26 m. Rising is ~0.3 s of a 10 s episode, so the 15% height-credit floor
and the unsupported-ascent cost were cheaper than a slower two-foot rise.

- `style_one_foot_rise` (−12): one-foot load while rising, from a base height of 0.10 m until standing.
- Height-credit floor 0.15 → 0.05.
- Monitors `rise_two_foot_frac` / `rise_one_foot_frac`: share of rising steps (h ≥ 0.10, before first
  standing) with both / exactly one foot above 10% of the weight.

## Revision 6: COM centred between the feet before rising; slow motion

- `ready` also multiplies `exp(−(d_centre/0.03)²)`, where `d_centre` is the COM distance to the centroid of
  the loaded sole corners (the midpoint between the feet when both are down). Rising credit, the
  unsupported-ascent cost and the one-foot rise cost all follow from it: the robot must first centre the
  COM between both feet, then rise on both.
- `style_one_foot_rise` −12 → −20.
- Slow motion:
  - Motor torque–speed limit 6 → 4 rad/s (fade from 3 rad/s), 2 rad/s below the 6 rad/s hardware limit.
  - Overspeed cost from 3 rad/s (was 4), scale −1 → −2.
  - Velocity termination 12 → 5 rad/s (was 20 → 6).
  - `style_rise_speed` (−4): upward COM speed above 0.2 m/s, so a ~1.5 s rise is free and a ballistic
    0.5 s rise is not.


## Revision 13: planted soles, a centred torso and still standing (from scratch)

> **Lineage.** Revision 13 branches from revision 6 (`24fad42`), not from revision 12.
> Revisions 7–12 (feet-first gating, improper-rise termination, narrow-squat guide and squat starts) are
> superseded. Their code and notes remain in the history up to `5ecb878`. The training runs of this
> revision were named `rev6a`, its working name.

Rev 6 got up well, but rose on tilted feet (0% of rising steps on two flat soles) and, after resuming
past ~2000 iterations, oscillated at ~8.5 Hz while standing: joint-speed RMS 0.29 → 2.0 rad/s by
iteration 6300, as the action scale fell to 0.25. The only standing-motion cost was
`stable_standing` at −0.05, about −0.1 per episode against a return of ~350. Revision 13 starts
again from scratch with the changes below. HoST multi-critic PPO, the force and action-scale
curricula, and every other term are unchanged.

`cov_L`, `cov_R` are per-foot sole coverages. They are computed once per step in
`_update_stage_state`, and the override replaces host_safe_recovery's coverage everywhere, including
`loaded_sole_coverage_deficit`. Revision 13 defines them **geometrically**:

`cov = mean over the 10 strip end points (toe and heel of each of the 5 capsules) of exp(−(relu(z)/0.003)²) · gate`

- `z` is the point's height above the ground.
- `gate = 1 − exp(−relu(F − 0.01W)/0.05W)` is the foot-load gate.

**Why not the force-strip version:** the rigid capsules sink less than 1 mm. A rev6 standing rollout
(50k loaded-foot samples) gave:

| Measure | Value |
|---|---|
| Median sole roll | 1.04° |
| Force coverage (mean) | 0.21: one strip carries everything in 60% of samples |
| Force coverage ≥ 0.9 | 71% of samples below 0.1° roll, 0% above 0.25° |

That would have demanded sub-0.1° ankle-roll precision. With a 3 mm tolerance, coverage ≥ 0.9
corresponds to about ≤ 1.7° of roll or pitch. On the same rollout, standing coverage is 0.96.

- `coverage_pair = 0.3·mean(cov_L, cov_R) + 0.7·min(cov_L, cov_R)`.
- `support_gate = smoothstep(share₂) · max(sup_L, sup_R)`: a smooth gate on both feet carrying load.

| Term | Was | Now |
|---|---|---|
| `target_sole_contact_coverage` | +4, `0.5·mean + 0.5·cov_L·cov_R` | **+8**, `coverage_pair`. Contact-based, so it never restricts early recovery |
| `style_loaded_sole_coverage_deficit` | −8 | **−10**, same formula (load-weighted). `loaded_sole_tilt` stays −8 |
| `target_two_foot_flat_support` | +4 | **+6**, `share₂ · max(sup) · flat_L · flat_R · min(cov_L, cov_R)` |
| `target_torso_center_over_feet` | — | **+4**, `support_gate · lift · exp(−(e_lat/0.08)²)`. `e_lat = |(p_torso,xy − p_mid,xy)·lateral|`, with `p_mid` the mean sole centre and `lateral` the robot lateral axis (`geo.lateral_axis`) |
| `target_com_support` | +6 | +6, unchanged |
| `target_com_support_margin` | — | **+2**, `support_gate · lift · smoothstep(margin/0.025)`. `margin` is the signed COM depth inside the support polygon, so the bonus is full at ≥ 2.5 cm and 0 on or outside the edge |
| `target_stable_standing` | −0.05 | **−1.0**, same formula: standing gate · (mean q̇² + 0.25·|ω|²) · boost (≤ 2) |

| `style_loaded_foot_translation` | — | **−4**, `mean_feet(hold · relu(v_xy − 0.03)/0.10)` (see below). `loaded_sole_slip` stays −6 |

**`loaded_foot_translation`:**
- `v_xy` is the horizontal speed of the sole centre, finite-differenced over the policy step. It is the
  actual foot motion, not joint velocity.
- `hold` follows `support_memory` up (load must be sustained). While the sole is within 5 mm of the
  ground, it decays with τ = 0.25 s when the load drops. It resets to 0 once the sole is lifted.
- So dragging a foot after briefly unloading it, while it still touches the ground, still pays.
  Stepping and airborne repositioning are free.

**Measured on rev6 model_1200** (rollout, not training):

| Measure | Value |
|---|---|
| Gate on unloaded-but-touching moving steps | 0.31, vs 0.15 with `support_memory` alone |
| Gate on airborne moving steps | 0 |
| `loaded_foot_translation` cost | ≈ 0.09 per step |
| `loaded_sole_slip` cost | ≈ 0.02 per step |

The two terms overlap: both see a loaded sole that slides. They differ in two ways:
- **Velocity:** slip uses per-contact tangential velocity, weighted by each capsule's own load.
  Translation uses the foot as a whole, weighted by the held load.
- **What each catches:** translation also catches light-load shuffling.

Both are logged separately.

**Readiness:**
- Was: `feet_under_com · share₂ · min(flat) · com_score · exp(−(d_centre/0.03)²)`.
- Now: `ready = sqrt(coverage_pair · com_score) · support_gate`. The COM-centring factor is dropped;
  `ready_com_centre_sigma_m` is removed.
- Height credit (head height, target base height, foot-supported height): `0.05 + 0.95·ready` →
  **`0.2 + 0.3·support_gate + 0.5·ready`**.

**Intermediate steps from one foot to two.** The first 13 run, at `model_200`, rose on one foot in 48%
of rising steps and on two in 8%. Per rising step, style + target favoured two feet by only +0.14:

- The one-foot penalty was worth +0.195.
- A freshly loaded, tilted, close-placed second foot immediately paid stance width, coverage
  deficit, tilt, posture and translation, about −0.25 in total.
- The ready-based rewards paid almost nothing until both soles were planted exactly (ready 0.015).
- The height credit was 0.200 on one foot vs 0.212 on two.

Changes:

- **Height credit:** both feet carrying load already earns +0.3 (`support_gate`, smooth). Full
  readiness adds the remaining 0.5. Since `ready ≤ support_gate`, the steps are 0.2 (one foot) → 0.5
  (both loaded) → 1.0 (ready).
- **Settled load:** `loaded_sole_coverage_deficit` and `loaded_sole_tilt` are weighted by
  `sup · support_memory` instead of `sup`. A foot that has just touched down is not charged in full
  before it settles: `support_memory` rises with τ = 0.1 s.
- **Rising stance:** `stance_width` is ×0.25 while rising, ramping to ×1 with the standing gate. A
  second foot may land close to the first and be adjusted. `leg_contact` (−6) still charges legs
  that press on each other.

**Monitors** (`Episode_end/`, over standing steps):
- `standing_min_sole_coverage`;
- `standing_both_soles_cov90_frac` (both soles ≥ 0.9 coverage);
- `standing_torso_lateral_err_m`;
- `standing_com_margin_m` (signed, finite steps only).
- Over actuated steps:
  - `peak_hip_roll_rad` (max outward hip roll);
  - `loaded_foot_translation` (mean cost);
  - `loaded_drag_dist_left_m` / `_right_m` (horizontal sole travel while `hold > 0.5`);
  - `moving_contact_foot_load_w` (foot load / W while the foot moves in contact).

`config.json` records `staged.revision = "13"`.

### Revision 13, after standing: adjust the stance by stepping, return to the default place

`model_600` stood with its feet 9.8 cm outside the hips (default 5.7 cm) and slid them inward by about
4 cm per foot within the first second, mostly faster than 10 cm/s. It lifted a foot 0–0.15 times per
episode, and the feet ended staggered by about ±7.5 cm fore-aft.

Sliding paid off: the joint-space `default_posture` dropped by ~0.5, worth about +30 to +60 per
episode, against about −0.7 of slip and translation cost. Stepping cost `standing_load_imbalance`. At
the final action scale of 0.25, the same policy neither slid nor stepped: it stayed 13 cm wide.

**"Lifted" and "standing":**
- A foot is *lifted* when its sole clearance is ≥ 1 cm and its load is < 5% W.
- The robot is *standing* when base h > 0.30 m and upright > 0.8.

| Term | Scale | Value |
|---|---:|---|
| `style_standing_foot_anchor` | −4 | `standing_gate · mean_feet(hold · min(3, relu(|xy − anchor| − 1 cm)/2 cm))`. The anchor follows the foot while it is lifted or the robot is not yet standing, and freezes where the foot was set down. A planted foot that slides pays for as long as it stays displaced; lifting it re-anchors it for free |
| `target_stance_return` | +4 | `standing_gate · mean_feet(planted · ½[exp(−(e/2 cm)²) + exp(−(e/8 cm)²)])`, with `e` the horizontal distance of the sole centre from its default place relative to its own hip: 5.68 cm outward, 0.37 cm forward, measured at zero joints. Task space, so moving the joints without moving the foot does not earn it |

**Short steps while standing are exempt:**
- **Exempt terms:** `standing_load_imbalance`, and the swing leg's joints in `stable_standing`
  (base angular velocity still counts).
- **Who qualifies:** a lifted foot while the other is down, with `swing_credit(air, 0.3 s, 0.5 s)`.
- **Timer reset:** the air timer restarts only after the foot has been planted (≥ 15% W) for 0.25 s.
  Rapid taps do not reset it.
- **Budget:** 1.5 s of exemption per episode.

Both feet may step, one after the other.

**Measured on `model_700` (current behaviour, not trained with these terms):**

| Measure | Value |
|---|---|
| Anchor drift while standing | 3.6 cm (anchor −29 per episode) |
| Stance error | 8.8 cm (stance return +4.5 per episode) |
| Upper bound for stepping into place | about +30 return, no anchor cost |

**Monitors:**
- `standing_lifts_per_ep`, `standing_swing_exempt_s`;
- `standing_anchor_drift_m`;
- `standing_stance_offset_err_m`, `final_stance_offset_err_m`.

Trained from scratch: at iteration 700 the policy noise std had fallen from 0.8 to 0.15, too little
exploration to discover stepping.

### Revision 13, after standing (2): make stepping discoverable, make sliding escalate

**What happened at `model_800`** (first run with the anchor):
- After standing, the policy slid each loaded foot 8–12 cm in the first 3 s.
- It lifted a foot 0–0.27 times per episode, and reached a stance error of 3.7 cm.
- It accepted about −57 of anchor cost per episode. A single lift would have re-anchored the foot,
  but the policy had almost never lifted a foot while standing, so it never saw that.

The changes:

- **Stance starts.** 15% of training resets begin standing in a displaced stance, from an IK library
  of 64 poses (`stance_poses.py`):
  - Each foot is 2–8 cm wider than default; feet are mostly staggered up to ±8 cm fore-aft.
  - Soles flat, COM between the feet, torso upright, base 0.33–0.35 m, random yaw.
  - Episodes are actuated at once with no pull force. They are excluded from the curriculum EMA and
    from the fallen-start stage, rise and standing metrics.
  - Logged: `stance_start_success`, `stance_start_lifts_per_ep`, `stance_start_final_err_m`,
    `fallen_success`.
- **`target_step_placement` (+100, an impulse).** At each touchdown while standing:
  `clip((e_at_lift − e_at_touchdown)/2 cm, ±3)`. It is signed, so stepping out and back in nets zero.
  A step that removes 5 cm of stance error earns about +5.
- **Escalating anchor cost.**
  - The cap goes from 3 to 10 per foot.
  - The cost is multiplied by `1 + post_stand_drag/5 cm`. `post_stand_drag` is the episode's loaded
    sliding while fully standing, per foot max: travel of a foot within 5 mm of the ground carrying
    ≥ 5% W, counted from the first contact with no settling delay.
- **Drag termination (training only).** An episode ends when `post_stand_drag` exceeds a limit. The
  limit is off until the success curriculum starts, then goes from 20 cm to 6 cm as it completes.
  Logged: `term_drag`, `drag_termination_limit`, `post_stand_drag_m`.
- **Anchors** are initialised at the feet on the first step of every episode, so stance starts do not
  inherit a stale anchor.

Evaluation and playback disable the drag termination and the stance starts.

**Checked on `model_800`, all resets forced to stance starts:**

| Measure | Value |
|---|---|
| Standing at step 0 | yes, h 0.341, both feet loaded after one step |
| Stance error at reset | 5.8 cm |
| Feet slide (old policy) | 6.6 cm, `post_stand_drag` 7.4 cm |
| Anchor cost | about ×2.5 |
| Lifts | 0 |

### Revision 13, after standing (3): no drag termination, action-scale floor 0.30

Run `20261013_123752_rev6a` collapsed after iteration ~1200:

| Iteration | 1100 | 1600 | ~1635 |
|---|---:|---:|---:|
| Fallen-start success | 0.95 | 0.51 | 0.06 |
| Action scale | 0.47 | 0.26 | 0.24 |

- Once the success curriculum started, the drag termination (limit about 16–20 cm) ended 40–70% of
  episodes, together with the tightening velocity termination (up to 37%). The policy had not yet
  learned to step, so standing up became the risky action.
- Meanwhile the inherited action-scale schedule kept lowering the scale on head height alone.

Changes:
- `drag_termination = False`.
- `action_scale_floor = 0.30`: `update_force_curriculum` clamps `action_rescale` from below.

Training resumes from `model_1100` of that run (fallen success 0.95, stance error ~2 cm, no jitter).
The anchor escalation, step placement reward and 15% stance starts stay.
