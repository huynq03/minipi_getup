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
| 3 COM transfer | `target_com_support` | +6 | `com_score · share₂ · min(flat) · lift` | both feet loaded, torso up |
| 4 rise | height credit | × | head height (task), `target_base_height`, `foot_supported_height` × `0.05 + 0.95·ready`, `ready = feet_under_com · share₂ · min(flat) · com_score · exp(−(d_centre/0.03)²)` | always; rising pays fully only after feet are under the COM, both feet are down and the COM is supported; floor 0.05 keeps an early gradient |
| 4 rise | `style_one_foot_rise` | −20 | `max(sup_L, sup_R) · (1 − share₂·min(flat)) · rise_gate`, `rise_gate = σ((h − 0.10)/0.02) · (1 − standing gate)` | actuated; knee/body support with no foot load is not charged here |
| 4 rise | `style_unsupported_ascent` | −6 | `(relu(v_z − 0.05)/0.20)² · (1 − ready) · lift` | after actuation starts |
| 5 standing | `style_standing_knee_extension` | −16 | mean over legs of `standing gate · (relu(−q_calf − 0.25)/0.40)²`: 0 above ~22° flexion, 1.0 at the straight leg, 31.6 at the −2.5 reverse limit | standing only: straight or reversed knees are free while getting up but must return toward the flexed default once standing (static standing at calf 0 needs ~0.6 Nm at the knee) |
| 2 feet down / 5 | `style_leg_contact` | −6 | `sup_leg(‖F_left-leg↔right-leg‖) · lift`, with `sup_leg` using deadband 0.005W and scale 0.05W (~3.4 N) | after actuation, torso up |
| 5 standing | `style_standing_load_imbalance` | −6 | `((F_L − F_R)/(F_L + F_R + 0.1W))² · standing gate` | standing; ≈ 0.64 for a 64/4 N split |
| knee support | `style_knee_load_support` | −45 × ramp | the safe-recovery knee-housing load cost × `0.25 + 0.75·clip(t_knee/0.8 s)` | `t_knee` grows while knee support > 0.1 and decays with τ = 0.3 s otherwise |
| 4 rise | `style_rise_speed` | −15 | `min((relu(v_z,COM − 0.20)/0.10)², 9)` | actuated; a slow rise (~0.17 m/s) is free |
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

## Revision 7: rising on two flat feet is enforced

rev-6 `model_600` from supine: the right leg splayed out (hip roll −1.33) and pushed, while the left knee
was reversed (−1.2) and the left foot propped on its edge/toe with the sole centre ~8 cm up. The force-only
two-foot share counted that as two feet, and the rise lasts only 0.15–0.3 s, so per-step penalties were
cheaper than standing up early (COM rose at 0.5–0.75 m/s).

- `share₂·min(flat_L, flat_R)` (two loaded *flat* feet) replaces the force-only share in
  `style_one_foot_rise` and `target_com_support`.
- **Improper-rise termination** (training only, `improper_rise_termination`): before the first stand,
  with the base above 0.18 m and the torso up (`−g_z > 0.5`; revision 8: base above 0.12 m **or** torso
  more than 45° up), the episode ends after N consecutive steps
  with `share₂·min(flat)` below a minimum or the COM too far outside the support polygon. It switches on
  with the penalty curriculum (strength s > 0), like the velocity termination, so the robot first learns
  to stand at all, and tightens with s: minimum 0.15 → 0.5, COM 10 → 3 cm outside, N 15 → 5 steps
  (0.3 → 0.1 s). With the final thresholds from the start, every rev-6 `model_600` episode in all five
  orientations was cut (0.55–0.8 s after actuation), leaving no learning signal. Upright kneeling (base ≈ 0.22 m) is cut by it too. Logged as
  `term_improper_rise`. Evaluation and playback disable it.
- `style_rise_speed` −4 → −15.

## Revision 8: feet first in every orientation

rev-6 `model_700` raised and straightened the torso before the feet were down: from supine the torso was
upright (−g_z 0.92) at 0.30 s with no foot on the ground, and two flat feet carried the load only from
~0.66 s at a base height of 0.325 m. Whenever both feet were loaded during the rise, the more tilted foot
was at a median 67° (edge/toe). The largest reward, the task orientation term (~40 per episode), paid for
an upright torso whatever supported it, and every feet-related cost started at 0.10–0.14 m.

- **A, feet-gated torso rewards:** `task_orientation`, `target_target_orientation` and
  `style_ground_parallel` × `0.1 + 0.9·share₂·min(flat)`.
- **B, `style_torso_lift_without_feet` (−10):**
  `min((relu(v_z,base − 0.02)/0.10)² + (relu(d(−g_z)/dt − 0.2)/1.0)², 9) · (1 − share₂·min(flat))`, from
  actuation until the first stand, at any height (sitting up counts).
- **C, earlier improper-rise check:** base above 0.12 m (was 0.18) or torso more than 45° up
  (`−g_z > 0.707`).
- Prone and side resets cannot put the soles down while lying (no arms); they pay B while rolling or
  kneeling the torso up, which pushes them to get the feet under the body first.

## Revision 9: feet first only above sitting/kneeling height

rev-8 `model_500` never stood unassisted (0/500); 45–70% of episodes ended kneeling upright (base
0.10–0.13 m, knees loaded, no flat foot) with the hips pressed on their inner limits. Moving from sitting
or kneeling onto two flat feet always lifts the hips while the feet are not yet flat, and prone/side
resets must raise the torso before the feet can get under the body; A and B charged exactly those
transitions, so kneeling (10% orientation credit, ~−0.9 knee cost) was the cheapest place to stay. C never
switched on because the two-foot success EMA stayed 0.

- A: the feet credit applies only above a base height of 0.16 m (σ width 0.02); sitting or kneeling upright
  is paid in full.
- B: only base-height increase (no uprightness rate), only above 0.14 m.
- C: base above 0.18 m only (no torso-angle trigger).
- Flat-feet definition, one-foot rise (−20), rise speed (−15) and the curriculum-scaled termination stay.

## Revision 10: feet-gated orientation (A) removed

rev 9 never stood in 316 iterations, even with the pull force (rev 8: 0.84, rev 6: 0.98 at the same
point). Sitting or kneeling upright below 0.16 m earned the full task orientation reward (~43 per
episode, as much as standing in rev 6), and rising above 0.16 m without two flat feet cut it to 10%, so
every way up lost reward. A is disabled (`orientation_feet_credit_floor = 1.0`); the orientation reward is
as in rev 6. Feet first is enforced by B (base lift above 0.14 m without two flat feet) and C (improper-rise
termination above 0.18 m with the curriculum), which charge *how* the robot rises without making rising
itself unprofitable.

