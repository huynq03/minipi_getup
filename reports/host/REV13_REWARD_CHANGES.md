# Revision 13 (working name 6a): reward changes from rev 6, by HoST reward group

| | |
|---|---|
| Base | rev 6, commit `24fad42`, run `20261012_201113_rev6_centred_slow` |
| Current code | commit `ea5e80f`, branch `host-staged-rev6-resume`, worktree `/home/huy/minipi_getup_host_rev6` |
| Unchanged | HoST multi-critic PPO, group weights task 2.5, regu 0.1, style 1, target 1, original HoST files (`src/minipi_getup/host/*`) |
| Scale convention | HoST multiplies every scale by dt = 0.02 |
| Full equations | `src/minipi_getup/host_staged_recovery/REWARDS.md`, section "Revision 13" |

## Goals

1. Rise on two flat, fully planted soles, with the torso and COM centred over the feet.
2. Stand still. Rev 6 resumed to iteration 6300 shook at about 8.5 Hz: joint-speed RMS went from 0.29 to 2.0 rad/s.
3. Stop dragging loaded feet. After standing, reposition the feet by lifting them and stepping back to
   the default place under the hips.

## Shared quantities (new or redefined)

| Quantity | Definition | Why |
|---|---|---|
| `cov_L`, `cov_R` (sole coverage) | **Geometric**. Mean over the 10 strip end points of each sole (toe and heel of the 5 capsules) of `exp(−(relu(z)/3 mm)²)`, times a load gate `1 − exp(−relu(F − 0.01W)/0.05W)`. Overrides `_sole_coverage`, so every term that uses coverage changes. | The old force-strip coverage needed sole roll below 0.1° to reach ≥ 0.9. The rigid capsules sink < 1 mm. At rev 6's median roll of 1.04°, one strip carried all the load and coverage was 0.21. Geometric coverage ≥ 0.9 corresponds to about ≤ 1.7° of roll or pitch. |
| `coverage_pair` | `0.3·mean(cov_L, cov_R) + 0.7·min(cov_L, cov_R)` | Mostly the worse sole. |
| `support_gate` | `smoothstep(share₂) · max(sup_L, sup_R)` | Smooth "both feet carry load" gate. |
| `ready` | `sqrt(coverage_pair · com_score) · support_gate`. Was `feet_under_com · share₂ · min(flat) · com_score · exp(−(d_centre/3 cm)²)`. | Requested redefinition. COM centring is dropped from readiness; torso centring is rewarded separately (target). |
| Stance error `e` | Horizontal distance of each sole centre from its default place relative to its own hip: 5.68 cm outward, 0.37 cm forward, measured at zero joints. | Task-space target for the feet. |
| Lifted foot | Sole clearance ≥ 1 cm and load < 5% W. | Defines a real step. |
| Standing | Base height > 0.30 m and upright > 0.8. | When anchors freeze. |

## Task group (×2.5)

No term added or removed. `orientation × head_height` is unchanged. Only the height credit it is
multiplied by changes. The same credit also multiplies `target_base_height` and `target_foot_supported_height`.

| | Rev 6 | Rev 13 |
|---|---|---|
| Height credit | `0.05 + 0.95·ready` | **`0.2 + 0.3·support_gate + 0.5·ready`** |

**Why.** The first 13 run rose on one foot in 48% of rising steps and on two in 8%. At `model_200` the
credit was 0.200 on one foot and 0.212 on two, so the largest critic saw no reason to load a second
foot. With the middle step, loading both feet already raises the credit, before the soles are planted
exactly. On the same policy, the credit became 0.20 on one foot vs 0.43 on two.

## Regu group (×0.1)

No changes. `action_rate` −0.03, `smoothness` −0.03, `dof_pos_limits` −100 and the others are as in rev 6.

## Style group (×1)

| Term | Rev 6 | Rev 13 | Change |
|---|---:|---:|---|
| `loaded_sole_coverage_deficit` | −8 | **−10** | Uses geometric coverage. Weighted by **settled** load, `sup · support_memory`, instead of `sup`, so a foot that has just touched down is not charged in full (`support_memory` rises with τ = 0.1 s). |
| `loaded_sole_tilt` | −8 | −8 | Same scale. Also weighted by settled load. |
| `stance_width` | −4 | −4 | **×0.25 while rising**, ramping to ×1 with the standing gate. A second foot may land close to the first and be adjusted later. `leg_contact` −6 still charges legs pressing together. |
| `standing_load_imbalance` | −6 | −6 | **Exempt during a short step**: one foot lifted, the other planted. Exemption lasts 0.3 s per lift, fading out by 0.5 s. The timer restarts only after a 0.25 s touchdown at ≥ 15% W. Budget 1.5 s per episode. |
| **`loaded_foot_translation`** (new) | — | **−4** | `mean_feet(hold · relu(v_xy − 0.03)/0.10)`. `v_xy` is the sole-centre horizontal speed. `hold` follows `support_memory` and, while the sole is within 5 mm of the ground, decays with τ = 0.25 s after unloading; it resets to 0 once the foot is lifted. So unloading a foot briefly while it still touches the ground does not avoid the cost, but stepping is free. Rollout check: the gate on unloaded-but-touching moving steps was 0.31, vs 0.15 with `support_memory` alone; airborne steps 0. |
| **`standing_foot_anchor`** (new) | — | **−4** | `standing_gate · mean_feet(hold · min(10, relu(drift − 1 cm)/2 cm)) · (1 + post_stand_drag/5 cm)`. `drift` is the distance of a planted foot from where it was set down. The anchor follows the foot while it is lifted or the robot is not yet standing, and is initialised at the feet each episode. `post_stand_drag` is the episode's loaded sliding while standing, so the cost **escalates** the longer the robot keeps sliding. |
| `loaded_sole_slip` | −6 | −6 | Unchanged. Logged separately from translation. |

## Target group (×1)

| Term | Rev 6 | Rev 13 | Definition |
|---|---:|---:|---|
| `sole_contact_coverage` | +4 | **+8** | `coverage_pair`. Was `0.5·mean + 0.5·cov_L·cov_R`. Contact-based, so it does not restrict early recovery. |
| `two_foot_flat_support` | +4 | **+6** | `share₂ · max(sup) · flat_L · flat_R · min(cov_L, cov_R)` |
| **`torso_center_over_feet`** (new) | — | **+4** | `support_gate · lift · exp(−(e_lat/8 cm)²)`. `e_lat` is the torso's offset from the feet midpoint along the robot lateral axis. |
| `com_support` | +6 | +6 | Unchanged. |
| **`com_support_margin`** (new) | — | **+2** | `support_gate · lift · smoothstep(margin/2.5 cm)`. `margin` is the signed depth of the COM inside the two-foot support polygon. |
| `stable_standing` | −0.05 | **−1.0** | Same motion cost (mean q̇² + 0.25·\|ω\|², standing gate, boost ≤ 2), but the joints of a leg in an exempt short step are left out. |
| **`stance_return`** (new) | — | **+4** | `standing_gate · mean_feet(planted · ½[exp(−(e/2 cm)²) + exp(−(e/8 cm)²)])`. Fine and coarse kernels: the coarse one gives a gradient from the 8.8 cm staggered stance the policy stood in. |
| **`step_placement`** (new) | — | **+100** (impulse) | At each touchdown while standing: `clip((e_at_lift − e_at_touchdown)/2 cm, ±3)`. Signed, so stepping out and back in nets zero. A step that removes 5 cm of stance error earns about +5. |

## Outside the reward groups

| Mechanism | State | Notes |
|---|---|---|
| Stance starts | **On, 15%** | Episodes begin standing in a displaced stance, from an IK library of 64 poses (`stance_poses.py`): each foot 2–8 cm wider, staggered up to ±8 cm, soles flat, COM centred. They are actuated at once with no pull force. They are excluded from the curriculum EMA (new `_penalty_ema_ids` hook in `host_safe_recovery`) and from the fallen-start metrics. |
| Drag termination | **Off** | Ends the episode when post-stand loaded sliding exceeds a curriculum limit (20 → 6 cm). It was turned off after collapsing run `20261013_123752`: once active it ended 40–70% of episodes, and fallen-start success fell from 0.95 to 0.06. |
| Action-scale floor | **0.30** | The inherited schedule lowers the scale to 0.25 on head height alone. At 0.24 the same run could no longer stand up or step. |
| `--resume` | Added | `train.py` restores the policy, optimizer and curriculum state from a checkpoint. |

Evaluation and playback turn off the velocity termination, the drag termination and the stance starts.

## Run history

| Run | Change | Result |
|---|---|---|
| `20261013_094633_rev6a_v1_onefoot` | First 13 (coverage, torso, margin, stable standing, translation) | Rose on one foot: 48% one-foot vs 8% two-foot rising (model_200). |
| `20261013_101113_rev6a_v2_drag` | + intermediate height-credit step, settled-load sole costs, rising stance relief | model_500: standing coverage 1.00, torso 2.5 cm off-centre, COM margin 8 cm, joint speed 0.01 rad/s (no jitter). Rose on two flat feet 22–60% of the time, except prone (4%). Feet dragged ~10 cm while rising, ~4 cm after standing; at scale 0.25 the stance stayed 13 cm wide. |
| `20261013_112654_rev6a_v3_anchor` | + anchor, stance return, step exemption | model_800: stance error 3.7 cm, but feet slid 8–12 cm after standing and almost never lifted (anchor cost −57 per episode, accepted). |
| `20261013_123752_rev6a_v4_dragterm` | + stance starts, step reward, escalating anchor, drag termination | model_500 lifted feet in stance starts (~10 per episode) but still slid. model_1100: fallen success 0.95, stance error ~2 cm. It then collapsed after iteration 1200 (drag termination plus action scale 0.24). Stopped at 1694. |
| (next) | drag termination off, action-scale floor 0.30 | Code ready (`ea5e80f`), **not trained**. Launcher: `scripts/train_rev6a_resume.sh` (resumes model_1100). |

## Open issues

- **The policy has not yet learned to step.** It closes the stance by sliding once after standing,
  then stands still. Stepping appeared at `v4/model_500` and vanished as the action scale fell.
- **Prone rarely rises on two flat feet** (0–8%).
- **Hip roll opens wide** (about 1.6–1.9 rad) at the sit-to-rise transition, to plant both feet early.

## Videos

In `reports/host/`, one combined file per checkpoint plus one file per fall orientation. Each shows
the median episode by time to stand, at the training action scale, with no pull force.

- `20261013_123752_rev6a_model_1100*.mp4`
- `20261013_123752_rev6a_model_1200*.mp4`
