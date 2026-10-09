# Recommendation: reducing Mini-Pi recovery joint speeds

This builds on `summary.md`. Every claim below points to a measurement there. Nothing has been changed or trained yet.

## The measured problem

1. **Large target jumps cause the spikes.** Over 99 % of \|qd\| > 20 rad/s events follow a ≥ 0.25 rad, 20 ms step of that joint's PD target toward the motion. The spike probability rises from 1.9 % (step < 0.05 rad) to 61 % (step ≥ 1 rad).
   - With a large error the PD acts as a velocity servo, qdot ≈ (kp/kd)·error: 19–50 s⁻¹ depending on the joint.
   - Nothing in the plant limits that speed: armature 0, no torque-speed derating, no slew. Free-limb inertia of 0.0004–0.004 kg m² reaches 20 rad/s in 0.5–5 ms at 16 Nm.
2. **The spikes sit in the get-up.** Phases A–C take 0.3–0.6 s of each episode. Phase A alone holds 48–54 % of the \|qd\| > 10 rad/s samples, and the first 100 ms after the passive window holds 34–42 % of the > 20 rad/s events. Standing (D) is moderate: p99 4–7 rad/s.
3. **The velocity penalty cannot compete.**
   - It is sampled at 50 Hz and misses 26–43 % of over-limit physics steps.
   - One full get-up costs about 1.6 reward of `qd_soft_envelope`, while each second on the ground costs about 35 relative to standing.
   - Training keeps speeding up the get-up: phase-A share of samples above 10 rad/s rose from 15 % to 28 % between iterations 3000 and 4500.
4. **There are no explosions** in 1024 evaluation episodes. Physics instability is not the source.
5. **Joint-limit violations are a separate problem** (up to 0.19 rad, mostly while standing). The policy presses targets into the range clip, and the soft limit constraint gives way.

## Options compared

| | A. Raise the velocity-penalty weight | B. Lower the penalty threshold | C. Progressive penalty curriculum | D. Joint-target rate limiter | E. Physics / contact stability | F. Per-joint limits |
|---|---|---|---|---|---|---|
| **Problem it addresses** | indirectly: the speed after it happens | same as A, plus moderate speeds (4–6 rad/s) | same as A, phased in | directly: the target step that causes 99 % of spikes (finding 1) | explosions, which are absent (finding 4) | per-joint differences: ankle roll inertia ≈ 0.0004, hip pitch / calf dominate the spikes |
| **Expected effect on recovery** | uncertain. The optimum get-up time scales roughly as √weight, so a noticeable change needs about 10–100× the weight, and the penalty would then also dominate the lifting phase where speed is needed. Risk of slipping back to the 2500-type plateau (24–44 %) | lowering to 4 rad/s would also penalize standing (D p99 4–7 rad/s) and the needed motion | better than A or B (gradual), same final trade-off | **measured zero-shot: 94 % at 0.3 rad/step** (100 / 88 / 91 / 97 %); expected to return to ~100 % after fine-tuning | none expected | depends on the values; uniform 0.3 is already measured |
| **Expected effect on joint speed** | lowers sampled speeds. Spikes inside a 20 ms step are partly invisible (26–43 % missed), and peaks form within 5–20 ms of a target step | as A | as A | **measured zero-shot: > 20 rad/s events −96 %**; p99 A / B / C drops from 27 / 25 / 20 to 13.5 / 14.3 / 14.4; max 47 → 35 | none on spikes (they occur in valid motion) | finer control of the ankle and hip pitch |
| **Drawbacks** | reward retuning; slower convergence; does not bound peaks; sampling bug as long as it stays at 50 Hz | as A | more tuning surface; still does not bound peaks | the target lags the action, a small partial-observability change (the policy sees its action, not q*). The get-up is slower (median time to stable 0.58 → 2.8 s zero-shot), and the fraction above 6.28 rad/s rises because the get-up lasts longer. The old 0.06 rad/step slew of v0–v2 coincided with a stalled h1 → h2 transition (v2 best about 13 %), but that run also had a torque-speed envelope, so the two effects are not separated. **At 0.2 rad/step zero-shot recovery is only 8 %.** | adding armature or a stiffer limit is a plant change that needs motor data we do not have (rotor inertia, gear ratio). Guessing values would make the sim less faithful, not more | more parameters to choose without data |
| **Fine-tune from model_3000?** | yes | yes | yes | **yes**: 94 % retained zero-shot | n/a | yes |
| **Changes the plant / action contract?** | no (reward only) | no | no | **yes**: deploy must apply the identical limiter | yes (plant) | yes (if part of D) |

### Two side notes, not primary

- **Evaluate `qd_soft_envelope` at every physics step**, e.g. as the maximum over the step. This makes the existing term see what it is meant to penalize, a 1.3–1.5× larger value. It changes the objective, though, so it is not "zero-change". It is a reasonable companion to D, not a replacement.
- **Joint-limit pressing in standing is a separate follow-up.** Options include clipping targets to the range minus a margin (an action-contract change) or a stiffer limit `solref` (a plant change).
  - The rate limiter does not address it.
  - Mixing it into the same experiment would confound the attribution.

## Primary recommendation: one experiment

**Fine-tune model_3000 with a PD-target rate limiter of 0.3 rad per policy step. Change nothing else.**

### Specification

**Limiter**
- q*ₜ = q*ₜ₋₁ + clip(clip(q_default + scale·clip(a, ±50), q_min, q_max) − q*ₜ₋₁, −0.3, +0.3).
- At the end of the passive window, q*ₜ₋₁ is the measured pose. This is exactly the counterfactual that was evaluated.
- It belongs in `FtsrJointAction.process_actions`, and must be implemented identically in deploy.
- Observations are unchanged. The policy still sees its raw action as `last_action`.

**Initialization**
- Start from `model_3000.pt` (sha256 5ccf02e1…), the best autonomous checkpoint (100 % in this audit, 99.6 % in the periodic eval).
- It is preferred over 4500 because training after 3000 kept making the get-up faster and its speeds higher.
- Load weights plus optimizer state. Assistance stays off: t > t_tag, unchanged schedule.

**Kept unchanged:** the 16 Nm cap, PD gains, rewards (including `qd_soft_envelope` at 6.28 / −0.01), stages, PPO, poses, passive window and physics.

**Length:** 1000–1500 iterations. Use the periodic no-assist eval at every 250 iterations, plus this audit (`velocity_audit rollout/analyze`) at the end.

**Acceptance**
- No-assist recovery ≥ 95 % in every pose (64 envs per pose, seed 2150).
- \|qd\| > 20 rad/s events per episode ≤ 10 % of model_3000's (≤ 1.7 per episode).
- p99 \|qd\| in phases A–C ≤ 15 rad/s.
- No new limit-violation regression.

**Only after acceptance:** tighten to 0.2 rad/step in a second fine-tune and measure again. Zero-shot 0.2 gives 8 %, so it must be reached by adaptation, not by a jump.

### Why this one

- **It targets the cause measured in this audit.** Large target steps are followed by PD-servo speeds that nothing in the plant limits. It bounds the requested target speed (15 rad/s) instead of relying on a 50 Hz penalty that the reward balance shows cannot compete.
- **It is the only option with direct evidence of preserving recovery:** 94 % zero-shot from the same checkpoint, with a 96 % reduction of > 20 rad/s events.
- **It is small.** One scalar in the action pipeline, no reward retuning, and fine-tuning instead of retraining.

### What it will not fix

- **Moderate speeds stay.** Phase p99 remains about 14 rad/s. Pushing below that needs 0.2 or a velocity term afterwards.
- **Joint-limit pressing while standing is untouched.**
- **Hardware feasibility is still unverified.** The real motors' no-load speed and rotor inertia are not known here; the earlier "H-conservative" 7.85 rad/s envelope was a hypothesis, not a measurement. A 15 rad/s target slew may still exceed the hardware. **Not hardware-ready.**

### Not necessary

A from-scratch retrain. Zero-shot retention (94 %) shows that the policy's strategy survives the limiter.
