# Mini-Pi deployment contract v2 (`mini_pi_fsm`, read-only audit)

Written 2026-10-07 from `/home/huy/Hightorque_Pi/mini_pi_fsm` HEAD `ddb7038` (nothing
modified). Paths relative to `mini_pi_fsm/deploy/` unless stated. Every item was
checked in the current code; nothing is taken from older notes.

| # | Item | Contract | Evidence |
|---|---|---|---|
| 1 | Joint order | Robot order: left leg then right leg (`l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch, l_ankle_roll, r_…`). Policy order is whatever `deploy.yaml joint_names` lists, resolved by name | `robots/mini_pi/config/mapping.yaml`, `doc/policy_format.md` |
| 2 | Signs | `direction = [1,1,−1,−1,1,1, −1,1,−1,1,−1,1]` motor→robot; robot frame = URDF frame | `mapping.yaml`, `JointMapper.h` |
| 3 | Offsets | `joint_offset = [−0.25,0,0,0.65,−0.40,0]×2`; `robot_q = dir·motor_q[map] − offset` | `mapping.yaml` |
| 4 | Joint limits | Safety `q_lower/q_upper` ±3.14 (deliberately loose, "REQUIRES HARDWARE TEST"): **not physical joint protection**. Physical ranges = URDF/XML. Invariant for the port: `q_target` clipped to the physical per-joint ranges (or documented margins) identically in training and in `deploy.yaml` `clip` | `safety.yaml`, `joint_actions.h` |
| 5 | Policy rate | `step_dt` from `deploy.yaml` (0.02 in every package); fixed-grid policy thread | `RLPolicyRunner.cpp:369` |
| 6 | Low-level rate | `control_hz: 1000` command refresh; motor feedback query every 2 ms; firmware PD rate itself undocumented. Simulation uses 500 Hz PD: an approximation that needs sensitivity validation | `robot.yaml` |
| 7 | PD on hardware | per package `stiffness`/`damping`; motor firmware PD | `RLPolicyRunner.cpp:280–285`, `HighTorqueHardware.cpp:235` |
| 8 | Target hold | latest policy output re-sent every control tick (ZOH) | `State_RLBase::run` |
| 9 | Action scaling | per-joint `scale` (scalar or list) | `joint_actions.h` |
| 10 | Action clipping | optional `raw_clip [lo, hi]` on the network output | `joint_actions.h` |
| 11 | Target clipping | optional per-joint `clip [[lo,hi]×n]` on `q_target` | `joint_actions.h` |
| 12 | Slew / rate limit | **none today** (no action or target slew limiter exists). Required by the user's operational contract: a GetUp-state `q_target` slew limit of 0.06 rad per 20 ms step (3 rad/s), see below | `joint_actions.h`, `RLPolicyRunner.cpp` |
| 13 | Motor command | `pos_vel_tqe_kp_kd2(q*, dq=0, τ_ff=0, kp, kd)` | `HighTorqueHardware.cpp:235–242` |
| 14 | Software torque clamp | none (`enable_torque_limit: false`) | `safety.yaml` |
| 15 | Firmware torque limit | unknown; `tor_limit_enable: false` in the loaded robot param file; SDK exposes `motor_torque_limit_flag` | `12dof_STM32H730_pi_lubancat_params.yaml`, `HighTorqueHardware.cpp:280` |
| 16 | IMU signals | orientation quaternion (→ rpy, projected gravity after `imu.mount_rpy`), gyro | `observations.h`, `robot.yaml` |
| 17 | Angular velocity | `root_ang_vel_b` = IMU gyro, body frame, rad/s | `observations.h:62` |
| 18 | Projected gravity | `R_wbᵀ [0,0,−1]` from the IMU orientation | `observations.h:104` |
| 19 | Joint pos / vel | `joint_pos_rel = q − default`, `joint_vel` rad/s; scaling via per-term `scale`, optional `clip`, `scale_first` | `observations.h`, `observation_manager` |
| 20 | `last_action` | previous network output after `raw_clip`, before scale/offset; zeros at reset | `observations.h:178`, `joint_actions.h::reset` |
| 21 | History | per term `history_length`; group `use_gym_history: true` = frame-major `[frame oldest: t0 t1 …]…[frame newest]`, false = term-major; oldest → newest; `history_init: current` (default, repeat current obs) or `zero` | `observation_manager.h` |
| 22 | ONNX input names | one input per observation group, the group name = input name (single group → `obs`) | `algorithms.h`, `doc/policy_format.md` |
| 23 | ONNX input count | any number of float32 inputs, batch 1 or dynamic | `algorithms.h` |
| 24 | ONNX output | exactly one float32 output `[1, 12]` (raw action) | `algorithms.h` |
| 25 | FixStand | interpolates from the measured pose to `q = 0` in 3 s with kp 80, kd 1.1; RL entry refused until done | `fsm.yaml`, `State_FixStand.cpp` |
| 26 | Velocity (RL) | single `RLBase` state; `policy.yaml` binds ONE policy runner to ONE state (`policy.state: Velocity`); entered only from FixStand (`rl`) | `fsm.yaml`, `policy.yaml` |
| 27 | Safety | latched faults checked every tick after arming: hardware fault, NaN state/command, motor timeout (0.1 s), IMU timeout (0.2 s), command timeout (0.05 s), position limits, orientation | `Safety.cpp` |
| 28 | Orientation fault | `max(|roll|, |pitch|) > 1.0 rad` → latched fault in **every** state, including Passive; response = kp 0, kd 1 damping; cleared only by operator reset, and re-raised while the condition holds | `Safety.cpp:255`, `CtrlFSM.cpp:126` |
| 29 | Stale checks | motor freshness per joint, IMU age, command age; RL action timeout (3 × step_dt or `action_timeout_s` 0.1 s), first-output timeout 0.5 s | `Safety.cpp`, `State_RLBase.cpp` |
| 30 | Emergency | latched damping override, FSM to Passive; `hard_stop: false` | `safety.yaml` |

Extra findings:

- `gait_phase` advances its clock **each time the term is evaluated**
  (`env->global_phase += step_dt/period`). If it appears in two observation groups (e.g.
  history group and current group), it advances twice per step. A Mini-Pi package must
  use it in exactly one group, or the training clock must match.
- `base_lin_vel` is unavailable on hardware; `params: {fixed: [0,0,0]}` feeds constant
  zeros, which matches the release's `base_lin_vel·0` slot exactly.
- `velocity_commands` returns SI values clamped to `commands.base_velocity.ranges`; the
  release scales commands by `[2, 2, 0.25]`, expressible with per-term `scale`.
- `deploy.yaml` with `simulation_only: true` is refused on real hardware.
- `robot.yaml` currently has `dry_run: true`.

## Consequences for the get-up port

1. **Observation interface fits without C++**: actor `o_t` and the student history can
   be two ONNX inputs (two groups: history group with `use_gym_history: true`,
   `history_length: 5`, and a current-frame group). Per-joint scale, `raw_clip` and the
   per-joint physical target clip are expressible in `deploy.yaml`; the target clip must
   equal the physical joint ranges (or documented margins) used in training.
   **The action path is not**: the required slew limit has no runtime support yet
   (item 4 below).
2. **A lying robot cannot run any policy today**: the global orientation check latches
   a fault while lying (tilt ≈ π/2 > 1.0), and the single RL runner is bound to
   `Velocity`, entered only through FixStand. Running a get-up policy needs a C++
   change in `mini_pi_fsm`: a `GetUp` state with its own policy runner and a
   state-local orientation exemption (stale IMU/motor, NaN, joint and target bounds,
   timeout and operator passive kept), restoring normal orientation safety after an
   upright hold, then handing over to FixStand/Velocity. Not done now (training first;
   the user forbade edits in `/home/huy/Hightorque_Pi` until training succeeds).
3. **gait clock**: if a Mini-Pi-only gait phase is used in r_w, it must be in one group
   only.
4. **Required GetUp-state action slew limiter (to add to mini_pi_fsm before any real
   actuation; not implemented now):**
   - Order: raw network output → `raw_clip` → `q_default + scale·a` → physical
     per-joint target clip → slew limit `|q*_t − q*_{t−1}| ≤ 0.06 rad` → motor command.
   - The slew is applied in the runtime action term, **not** inside the ONNX: it needs
     the previous commanded target as state, which the stateless ONNX does not have.
   - `last_action` keeps its current deploy meaning (raw output after `raw_clip`,
     before the slew). Training must observe the same quantity.
   - `q*_{t−1}` initialization at GetUp entry: the target the motors are actually
     holding at entry (e.g. measured `q` clipped to the physical ranges when entering
     from a passive/damping state). Simulation must initialize it the same way at the
     end of its settle/unactuated phase.
   - GetUp-state velocity fault at `|q̇| > 4 rad/s` (initial, tunable conservatively);
     the existing `enable_velocity_limit`/`dq_max` check is global to all states and
     would also constrain walking, so a state-local check (or state-switched config)
     is needed.
   - These go with the GetUp state of item 2 in one isolated change, after training
     succeeds.
