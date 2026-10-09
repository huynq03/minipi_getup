# Recommendation: starting checkpoint and limit for the rate-limiter fine-tune

This builds on `summary.md`. Every number is zero-shot, from a deterministic student with zero
assistance, three seeds and 1536 episodes per row unless noted. Nothing has been trained.

## Selection against the criteria (limiter 0.3 enabled)

| Criterion | model_3000 | model_4000 | model_5000 | model_8000 |
|---|---|---|---|---|
| 1. Recovery ≥ 95 % on every seed and every pose | **fails**: 93.4 %, worst pose 88.3 % | **passes**: 98.2 %, worst pose / seed 96.9 % | **passes**: 98.4 %, worst pose / seed 96.1 % | **fails**: 50.7 %, prone 24 % |
| 2. Reduction of > 20 rad/s events | −94.8 % (0.90 per episode) | −93.8 % (0.91 per episode) | **−94.3 % (0.73 per episode, lowest)** | −89.8 % (1.51 per episode) |
| 3. Lifting-phase p99 / max (rad/s) | 13.4 / 31.8 | 15.2 / 28.7 | 15.0 / 28.5 | 15.2 / 27.3 |
| Overall qd max (rad/s) | 43.4 | 34.3 | **32.0** | 35.0 |
| 4. Maximum joint-limit violation (rad) | 0.585 | 0.440 | **0.293** | 0.323 |
| Episodes with a violation > 0.05 rad | 92 % | 87 % | **80 %** | 96 % |
| 5. Numerical instability | none | none | none | none |
| 6. Median / p95 recovery time (s) | 3.22 / 11.0 | **1.34** / 7.67 | 2.00 / **7.20** | 4.86 / 13.6 |
| Robustness to a tighter limit (0.25 / 0.20, seed 2150) | not run | **88.1 % / 70.9 %** | 60.4 % / 48.2 % | not run |

### Ruled out

- **model_8000** is the newest and reaches 99.9 % without the limiter, but it collapses with it.
  It stalls between h1 and h2 in 45 % of episodes, its policy std is 3.04, and its evaluations
  dipped to 54 % at iteration 7000 during training. It is not a good base for further adaptation.
- **model_3000** was the velocity audit's choice, but it now fails criterion 1 on every seed:
  92.8–93.8 %, worst pose 88–91 %. In the audit's single seed it looked like 94 %.
  - Its get-up is the slowest with the limiter: median 3.2 s, p95 11 s.
  - It has the largest violations: 0.59 rad.
  - Its only advantage, the lowest lifting p99 (13.4), comes from a slower, more hesitant lift
    that fails more often.

### model_4000 vs model_5000

Neither is clearly superior.

| | model_4000 | model_5000 |
|---|---|---|
| Recovery with the limiter | 98.2 % | 98.4 % |
| Worst pose (min over seeds) | 96.9 % | 96.1 % |
| Spike events per episode | 0.91 | 0.73 |
| qd max | 34.3 rad/s | 32.0 rad/s |
| Joint-violation maximum | 0.44 rad | 0.29 rad |
| Median recovery time | 1.34 s | 2.00 s |
| Recovery at a 0.25 rad/step limit | 88 % | 60 % |
| Recovery at a 0.20 rad/step limit | 71 % | 48 % |

- On recovery and worst pose they are tied, within seed noise: 96–98 %.
- **model_5000 is better on safety metrics:** fewer spike events, a lower qd max and a much lower
  violation maximum.
- **model_4000 is better on adaptability:**
  - Its get-up is faster: 1.34 s median against 2.00 s.
  - It loses far less recovery when the limit tightens to 0.25 or 0.20. Its strategy depends less
    on large instantaneous target jumps.
  - This matters because the plan is to tighten the limit after the first fine-tune.

## Recommendation

**Fine-tune both model_4000 and model_5000 in a controlled comparison.**

- Use a limiter of 0.3 rad/step in training. It must be the identical code to this evaluation,
  in `FtsrAction.process_actions`, applied after the physical clip and initialized from the
  measured pose at the first actuated step.
- Change nothing else: rewards, PPO, stages, the plant, the 16 Nm cap and the gains.
- Load the weights and optimizer state, as in the resume.
- Run the same number of iterations for each, e.g. 1000.
- Evaluate with this tool every 250 iterations: 3 seeds × 128 per pose, the same initial states.
- If only one run is affordable, **start from model_4000**, because of its tolerance to tighter
  limits. The value of model_5000's lower spike and violation numbers depends on whether
  fine-tuning closes model_4000's gap, which only the controlled pair answers.

Acceptance for the fine-tuned checkpoint (limiter on):

- Recovery ≥ 95 % on every pose and every seed. The target is to return to ≥ 99 %.
- Events above 20 rad/s ≤ 0.9 per episode, i.e. no worse than zero-shot. Lifting p99 ≤ 15 rad/s.
- Median recovery time ≤ 1.5 s.
- Joint-violation maximum no worse than zero-shot.

## Is 0.3 rad/step the right threshold?

**Yes, for the first fine-tune.**

- It is the tightest value in the scan that keeps ≥ 95 % zero-shot on the two viable checkpoints.
- It already delivers what the limiter can deliver:
  - −94 % events above 20 rad/s.
  - About −50 % lifting p99.
  - A phase p99 cap of about 15 rad/s, which equals the commanded 0.3 rad / 20 ms.
  - −30 % in the maximum.
- Zero-shot, 0.25 and 0.20 lose 12–52 points of recovery, so they should not be the first step.
- Tighten to 0.25 and then 0.2 only after the 0.3 fine-tune recovers ≥ 99 %. This needs a
  separate approval, and is best done as a curriculum, not a jump.

## What the limiter does not solve (independent of the checkpoint)

- **Residual peaks of 25–35 rad/s** come from contact impacts and inertial coupling, not from
  target jumps. A lower limit, or a velocity term evaluated at every physics step, would be
  needed. See the velocity audit, recommendation side note 1.
- **Joint-limit pressing.**
  - The target sits on a range bound in 58–67 % of actuated joint-steps.
  - The low-gain thigh (hip yaw) and ankle roll joints are pushed up to 0.3–0.6 rad past the
    soft limit under body load.
  - The limiter lengthens the loaded B/C phases and raises the maximum. A target margin or a
    stiffer limit is a separate decision.
- **Hardware feasibility is unverified.** 15 rad/s of target slew and 16 Nm peaks have not been
  checked against the real motors, whose no-load speed and rotor inertia are unknown. The
  URDF lists 21 rad/s and 21 Nm. **Not hardware-ready.**
- **The limiter changes the action contract.** Deploy must implement the identical limiter,
  including its initialization from the measured pose when the GetUp state starts.

**Stopping here.** Fine-tuning awaits approval.
