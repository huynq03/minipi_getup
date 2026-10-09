# Zero-shot limiter sweep of the fine-tuned model_4500 (no training)

| | |
|---|---|
| Checkpoint | `logs/rsl_rl/minipi_ftsr_ref/2026-10-09_17-45-30_ftsr_limit0p3_ft4000/model_4500.pt`, sha256 `aea114e1…e7f1ba`. Fine-tuned at 0.3 rad/step. |
| Protocol | Task `Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless` (play cfg) plus the evaluation-wrapper limiter, bit-exact to the training implementation (validate t33). Deterministic student, zero assistance, seed 2150, 128 episodes per pose. Identical initial states at every level (maximum \|Δqpos₀\| = 0). |
| Unchanged | Rewards, physics, gains, the 16 Nm cap, joint ranges. No target margin. |
| Data | `limit_comparison.csv`, `per_pose_results.csv`, `failure_diagnostics.json`. One figure, `plots/report.png`: base height of every episode. |
| Commands | `limiter_eval rollout --limit L` and `limiter_eval compare`. |

## Results

| Limit (rad/step) | 0.3 | 0.2 | 0.15 | 0.1 |
|---|---|---|---|---|
| Recovery S / P / L / R (%) | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 99.2 / 73.4 / 79.7 / 89.1 | 53.9 / 3.1 / 30.5 / 35.2 |
| Overall / strict / foot-only (%) | 100 / 100 / 100 | 100 / 99.8 / 100 | 85.4 / 84.8 / 85.4 | 30.7 / 30.1 / 30.7 |
| Recovery time median / p95 (s) | 0.50 / 1.08 | 1.00 / 5.14 | 1.52 / 11.4 | 8.40 / 14.4 |
| Rise attempts to stand, median / p90 (successes) | 1 / 1 | 1 / 4 | 2 / 10 | 3 / 7 |
| qd p99 / max, A (rad/s) | 15.7 / 23.4 | 11.4 / 19.2 | 9.2 / 19.8 | 6.6 / 28.5 |
| qd p99 / max, B | 15.8 / 22.8 | 11.7 / 19.7 | 9.7 / 30.1 | 6.9 / 29.4 |
| qd p99 / max, C | 14.3 / 26.7 | 11.4 / 23.9 | 9.6 / 30.9 | 6.3 / 25.3 |
| qd p99 / max, D | 6.4 / 20.5 | 5.5 / 21.0 | 5.2 / 24.5 | 5.0 / 27.6 |
| Events > 20 rad/s per episode | 0.23 | 0.01 | 0.05 | 0.93 |
| Torque saturation (joint-substeps) | 0.44 % | 0.19 % | 0.17 % | 0.22 % |
| Joint-limit penetration max / p99 (episodes) / episodes > 0.05 rad | 0.241 / 0.126 / 65 % | 0.177 / 0.133 / 67 % | 0.383 / 0.155 / 69 % | 0.216 / 0.172 / 58 % |
| Limiter active (actuated joint-steps) | 8.4 % | 13.8 % | 22.4 % | 38.3 % |
| Limiter active in A / B / C / D | 36 / 23 / 16 / 8 % | 42 / 31 / 27 / 12 % | 45 / 38 / 37 / 15 % | 44 / 44 / 38 / 17 % |
| Invalid / NaN episodes | 0 | 0 | 0 | 0 |

A "rise attempt" is one rise of the base above 0.15 m followed by a drop below 0.12 m.

Failures are listed in `per_pose_results.csv`:

- **0.15:** every failure reached h1 (75 episodes): 53 reached h1 but did not hold, 17 high-but-tilted, 5 fell after recovery.
- **0.1:** 355 failures: 205 reached h1 but did not hold, 126 partial lift, 17 fell after recovery, 5 high-but-tilted, 2 never lifted.

## Failure analysis: evidence, and what it does not show

Successes and failures over the get-up window (phases A–C):

| | 0.15 success | 0.15 failure | 0.1 success | 0.1 failure |
|---|---|---|---|---|
| Rise attempts, median | 2 | 18 | 3 | 14 |
| Peak base vz before h1, median (m/s) | 0.73 | 0.71 | 0.58 | 0.62 |
| Limiter active | 38 % | 39 % | 39 % | 45 % |
| Command gap \|q_cmd − q*\|, mean (rad) | 0.21 | 0.22 | 0.18 | 0.24 |
| PD tracking error \|q* − q\|, mean (rad) | 0.11 | 0.11 | 0.07 | 0.10 |
| Torque saturation | 0.42 % | 0.43 % | 0.02 % | 0.32 % |
| Steps with l_calf at the cap | 9.2 % | 9.3 % | 0.3 % | 3.6 % |

### What the data show

- **Failure mode: repeated rise-and-fall, not stalling on the ground.**
  - At 0.1, 64 % of failures reach h1, and failures make a median of 14 rise attempts.
  - At 0.15 every failure reaches h1.
  - The robot gets up to about h1 to h2, cannot hold, falls back and tries again. This is
    visible in `plots/report.png`.
- **Even the 0.1 successes are not a controlled slow get-up.**
  - Median 3 attempts. The "4.6 s to h1" of the successes is mostly failed attempts before
    the final rise.
  - This is the wait-and-retry pattern, not a 5 s controlled motion.
- **Torque saturation is not what separates success from failure.**
  - At 0.15, saturation is identical in successes and failures (0.42 / 0.43 %; l_calf 9.2 / 9.3 %).
  - At 0.1, saturation is low in both (≤ 0.32 %). It is higher in failures, but those
    episodes spend more time in repeated loaded attempts.
  - The PD tracks the limited target to 0.07–0.11 rad on average.
  - So there is **no evidence that missing torque is the main cause**. A local, short torque
    limit at particular instants is not excluded by these aggregate numbers.
- **The peak upward speed of the base before h1 does not separate success from failure.**
  At 0.1 it is 0.58 for successes against 0.62 m/s for failures; at 0.15, 0.73 against 0.71.
  The data therefore **do not support "lack of momentum" as the deciding factor**. It is not
  ruled out either: momentum at other moments, or in other directions, was not measured.
- **The policy's commands do not fit the limit.**
  - The limiter clips 39–45 % of joint-steps during the get-up at 0.15 and 0.1, against 29 %
    at 0.3.
  - Failures at 0.1 have a larger command gap (0.24 rad) and more clipping (45 %) than
    successes (0.18 rad, 39 %).
  - This is consistent with "the policy requests corrections faster than the target may move"
    (delayed reaction). It is a correlation, not a causal test.

### Not concluded

- **Whether a curriculum to 0.1 will succeed.**
- **Whether the robot needs momentum to stand up.**
- **Whether a 5 s controlled get-up is reachable with the limiter alone.**
  - One relevant observation: fine-tuning at 0.3 compressed the time again, from 1.32 s
    zero-shot to 0.48 s, because the reward still pays for standing sooner.
  - A tighter limiter may therefore end up as a faster use of the allowed slew, not a slow
    5 s motion. This is a hypothesis to test, not a result.

## Recommendation: first fine-tune level = 0.2 rad/step

- **Zero-shot at 0.2 already passes the selection rule.** It is 100 % in every pose, 99.8 %
  strict and 100 % foot-only. Successful get-ups are single rises (median 1 attempt), so the
  policy's strategy still works and fine-tuning starts from a healthy state.
- **Its movement and safety numbers are better than at 0.3:** events > 20 rad/s 0.01 against
  0.23 per episode, phase p99 about 11.5 against 15.7 rad/s, and maximum penetration 0.177
  against 0.241 rad.
- **0.15 is a riskier first step.** Prone drops to 73 %, and its failures are the
  rise-and-fall oscillation. Adapting from there starts with 15 % failures.
- **0.2 alone does not reach 5 s.** It is the safe first step of a curriculum
  (0.2 → 0.15 → 0.1), to be judged after each step.
- Whether the 5 s target needs a time-related change, i.e. a reward or a height reference,
  should be decided from the 0.2 and 0.15 fine-tune results, not from this sweep.

Nothing was trained, no target margin or reward was changed, and no checkpoint was modified.
