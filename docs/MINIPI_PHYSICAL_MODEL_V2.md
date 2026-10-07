# Mini-Pi physical model v2 (evidence audit)

Written 2026-10-07 before any code on `rebuild/ftsr-reference-minipi`. Physical
authority: (1) `/home/huy/Hightorque_Pi` code and configs (read-only, untouched),
(2) verified datasheets, (3) current simulation values.

## 1. Robot model

| Quantity | Value | Source |
|---|---|---|
| Model | `src/minipi_getup/asset_zoo/robots/hightorque_minipi/xmls/cl_pai.xml` | repo |
| Mass | 6.94 kg | MuJoCo `body_mass.sum()`; identical to vendor URDF `clpai_12dof_0905.urdf` (6.94 kg) |
| Joint names | `{r,l}_{hip_pitch, hip_roll, thigh, calf, ankle_pitch, ankle_roll}_joint` | XML = vendor URDF = `mapping.yaml` |
| Joint ranges | hip pitch [−1.25, 1.75]; hip roll R [−0.5, 0.12] / L [−0.12, 0.5]; thigh (yaw) R [−0.6, 0.3] / L [−0.3, 0.6]; calf (knee) [−0.65, 1.65]; ankle pitch [−0.5, 1.3]; ankle roll R [−0.3, 0.8] / L [−0.8, 0.3] | XML; identical to vendor URDF |
| Nominal stance | all joints 0 (knee bend baked into body frames); base height 0.345 m, CoM 0.223 m | XML, old audit (measured) |
| Landmarks | lying ≈ 0.085 m, lowest statically balanced crouch ≈ 0.19 m | old audit (measured) |
| Armature / joint damping / friction | 0 in the XML; the vendor URDF has no `<dynamics>` | XML, URDF |
| Physics | 2 ms, decimation 10 → 50 Hz policy | repo; the deploy runs 1 kHz (§3) |

Deploy coordinates: `robot_q = dir·motor_q[map] − offset` (`JointMapper.h`), and every
mjlab package uses `default_joint_pos = 0` in this frame, i.e. the simulator joint
coordinate equals the deploy robot coordinate. Joint signs and zero match by
construction (same joint names, same URDF frames). A hardware sign check is still a
deployment prerequisite (`doc/deployment.md`).

## 2. PD gains

| Source | kp (hip pitch, roll, yaw, knee, ankle pitch, ankle roll) | kd | Status |
|---|---|---|---|
| Vendor RL controller `clpai_12dof_0905/devel_config/config.yaml` `running_kp/kd` | 60, 40, 20, 60, 30, 10 | 2.4, 0.8, 0.4, 2.8, 1.6, 0.3 | hardware RL path; PD at `pd_ctrl_f: 1000` |
| mini_pi_fsm `mjlab47` package `deploy.yaml` | same | same | per-package, carried to the motors |
| Vendor `pai_control` (`pai_model.yaml`), FixStand in `fsm.yaml` | 80 all | 1.1 all | other controllers |
| minipi_getup `get_minipi_robot_cfg` | 0.7 × 60, … | same kd | **simulation-only choice**, not deployed anywhere |

Decision: the plant uses **60/40/20/60/30/10, kd 2.4/0.8/0.4/2.8/1.6/0.3**: the gains of
the vendor RL controller and of the existing mini_pi_fsm RL packages
(`HARDWARE_CONSTRAINT`). The 0.7 factor is dropped.

## 3. Low-level control on hardware

- `State_RLBase::run` copies the latest policy output into the command each control
  tick; `control_hz: 1000` (`robot.yaml`). The target is **held** (zero-order) between
  50 Hz policy updates.
- The command is `pos_vel_tqe_kp_kd2(q*, 0, 0, kp, kd)` per motor
  (`HighTorqueHardware.cpp:235`, `RLPolicyRunner.cpp:392–396`): the **motor firmware**
  computes `τ = kp(q* − q) − kd·q̇` with no velocity target and no feed-forward torque.
- **No software torque clamp** in mini_pi_fsm (`safety.yaml`: `enable_torque_limit:
  false`, `tau_max: []`); the robot param file sets `tor_limit_enable: false` for all 12
  motors. What the firmware limits internally is unknown (the SDK only exposes a
  `motor_torque_limit_flag`).
- Simulation equivalent: MuJoCo position actuator (PD at 500 Hz) with the
  target held for 10 physics steps. Equivalent up to the 1 kHz vs 500 Hz PD rate.

## 4. Motor data: CONTRADICTORY (blocker)

| Evidence | Says | Source |
|---|---|---|
| Robot param file actually loaded by `mini_pi_fsm.launch` | all 12 motors `type: "5047_36_2"` | `install/share/sim2real/robot_param/12dof_STM32H730_pi_lubancat_params.yaml` |
| SDK motor table | `5047_36_2` = "new 5047_36 (all current motors are the new model)", torque coefficient 0.8030; driver documented as **HTDW-5047-36-NE** | `livelybot_serial/include/hardware/motor.h:75,86`, `sim2real_sdk/include/motor/motor_base.h:129` |
| Public product page | joint modules **HTDW-5036-02-DNE / -CNE**, robot max joint torque 16 Nm; no speed or rated-torque numbers on the page | fetched 2026-10-07 |
| User-supplied datasheet summary (ChatGPT, source not shown) | HTDW-5036-02: rated 6 Nm @ 50 rpm (5.24 rad/s), **no-load 75 rpm = 7.85 rad/s**, locked-rotor 21 Nm, 36:1 | conversation, unverified |
| Vendor URDF | `effort="21"`, **`velocity="21"` rad/s** on every joint | `clpai_12dof_0905.urdf` |
| Vendor `pai_control` | `tau_limits` 16 Nm (6 Nm ankle roll) | `pai_model.yaml` |
| Vendor RL controller | `torque_limits` 35 Nm | `devel_config/config.yaml` |
| Old minipi_getup | 16 Nm cap (`MINIPI_EFFORT_LIMIT`) | repo |

Conflicts:

1. **Motor model**: HTDW-5047-36 (SDK, loaded config) vs HTDW-5036-02 (product page,
   the source of the user's numbers).
2. **No-load speed**: 7.85 rad/s (user summary) vs 21 rad/s (vendor URDF). Locked-rotor
   21 Nm agrees with the URDF effort, but the speed differs by 2.7×.
3. **Torque cap**: 16 Nm (product page, `pai_control`), 6 Nm on ankle roll
   (`pai_control`), 35 Nm (vendor RL), none (mini_pi_fsm, firmware flags off).

Why it blocks: the instructions require the final torque-speed model from walking
pretraining iteration 0. The archived `GetupDeploy` runs showed the result hinges on it:
with a 7.85 rad/s curve the prone recovery was never found (0 %), while the 21 rad/s
URDF value would make the curve nearly inactive at the speeds observed (≤ 17 rad/s).
Picking one would be inventing the plant.

Supporting evidence of feasibility on hardware: the vendor RL package ships scripted
get-up waypoint sequences for all four fallen poses
(`sim2real/way_point/waypoint_lower_body_{front,back,left,right}_down.boost`, 9–16
keyframes, motor space, up to ~2.4 rad between keyframes). Their timing lives in a
prebuilt binary and is not recoverable from source.

## 5. Actuator model plan (once §4 is resolved)

- Keep MuJoCo's **native position actuator** (stable at 2 ms with zero
  armature, as all previous Mini-Pi training showed).
- Add the torque-speed envelope on top of it without changing the integrator: per
  physics step, set each actuator's `forcerange` from the measured joint velocity,
  `τ_max(q̇) = min(τ_cap, τ_stall·(1 − |q̇|/ω_0))` (linear stall-to-no-load), with the
  quadrant rule of a DC motor (braking torque not reduced). If MjLab does not allow
  per-step `forcerange` writes on GPU, the fallback is mjlab's `DcMotorActuator`
  (explicit PD) **plus** a documented reflected-inertia armature randomized in DR.
- The archived explicit-PD runs needed armature 0.01 kg·m² only to stop chatter; that
  value is not physical evidence. Reflected rotor inertia (rotor J × 36²) is not in any
  local document.
- Rated torque (if verified) is monitored (time above rated), not enforced.

## 6. What is needed from the user

One of:

- The datasheet of the motor actually installed (**HTDW-5047-36-NE**, SDK type
  `5047_36_2`): stall/peak torque, no-load speed at the joint, rated point, and
  ideally rotor inertia; or
- Confirmation that the HTDW-5036-02 numbers (7.85 rad/s, 21 Nm) apply to the installed
  motors; or
- A bench measurement: no-load joint speed at full command, and the hardware torque cap
  that should be treated as instantaneous maximum (16 Nm vs 6 Nm on ankle roll).
