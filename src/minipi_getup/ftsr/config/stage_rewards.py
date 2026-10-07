"""Stage-wise reward table for Mini-Pi (paper Table II; stages r_u, r_s, r_w).

Weights are per second: mjlab multiplies each term by ``weight * step_dt``, as
getup_gym does. 0 marks an inactive term (the paper's "-").

Adaptations to Mini-Pi (see docs/FTSR_REPRODUCTION_AUDIT.md, Sec. 4):

- Kernel widths that are lengths scale with the robot. JiaRan's stance is 0.75 m,
  Mini-Pi's 0.345 m, so the height kernel exp(-8.3 dh^2) becomes
  exp(-8.3 (0.75/0.345)^2 dh^2) = exp(-39 dh^2).
- Linear-velocity kernel: exp(-25 dv^2) (std 0.2 m/s, as in the tuned Mini-Pi
  velocity task). The first pretraining (ftsr_pretrain_rw_v1) used the Froude-scaled
  exp(-18 dv^2). With it, standing still under a 0.2-0.3 m/s command kept half the
  tracking reward, and the policy stood instead of walking. The yaw-rate kernel keeps
  8.3: rad/s doesn't depend on size.
- Torque-type penalties (|tau|^2 and |qdot tau|^2) are multiplied by (6)^2 = 36.
  JiaRan's joints (kp ~100, action scale 0.6) run at roughly 6x Mini-Pi's torques in
  its 9 Nm envelope. Table II's 1e-6 would otherwise vanish on a 9 Nm robot, where on
  JiaRan it was a weak but present regularizer.
- "Feet dist." uses Mini-Pi's 0.16 m stance width, and its weight is scaled by the
  same 0.75/0.345 length ratio (-2 -> -4).
- "Wheel force" becomes foot support force (feet carrying the body weight).
- An upside-down indicator joins the |g_xy| orientation term (released code's
  ``pen_base_orientation_z_l2``): |g_xy| is 0 both upright and inverted.
- Walking adds legged-gait terms that the wheeled robot didn't need: feet air time,
  feet slip, and a gait-clock contact reward (``feet_gait``, with the clock in o_t).
"""

from minipi_getup.ftsr.config.robot import STANCE_HEIGHT

# Target base heights h1, h2, h3 (Sec. 5 of the audit).
STAGE_HEIGHTS = (0.19, 0.30, 0.335)

HEIGHT_KERNEL = 8.3 * (0.75 / STANCE_HEIGHT) ** 2
LIN_VEL_KERNEL = 1.0 / 0.2**2
ANG_VEL_KERNEL = 8.3
TORQUE_SCALE = 36.0
# Tracking rewards only count once the torso is up (released: 0.6 of 0.75 m).
TRACKING_MIN_HEIGHT = 0.8 * STANCE_HEIGHT

# name: (r_u, r_s, r_w)
STAGE_WEIGHTS: dict[str, tuple[float, float, float]] = {
  "track_lin_vel": (0.0, 0.0, 7.0),
  "track_ang_vel": (0.0, 0.0, 4.0),
  "orientation": (-2.6, -2.8, -10.0),
  "upside_down": (-0.6, -1.0, -1.0),
  "torques": (-1.0e-6 * TORQUE_SCALE,) * 3,
  "dof_acc": (-2.5e-8, -2.5e-6, -2.5e-6),
  "dof_vel": (0.0, 0.0, -0.01),
  "action_rate": (-0.06, -0.02, -0.08),
  "action_smoothness": (0.0, 0.0, -0.04),
  "dof_pos": (-0.06, -0.1, -0.12),
  "dof_energy": (0.0, 0.0, -1.0e-4 * TORQUE_SCALE),
  "base_height": (4.0, 6.0, 5.0),
  "termination": (-200.0, -100.0, -500.0),
  "feet_distance": (0.0, 0.0, -4.0),
  "leg_bias": (-2.0, -0.02, 0.0),
  "no_fly": (-2.0, 0.0, -0.2),
  "feet_support": (4.0, 0.0, 0.0),
  # 1.0 in ftsr_pretrain_rw_v1 (no stepping emerged); 2.0 since v2.
  "feet_air_time": (0.0, 0.0, 2.0),
  "feet_slip": (0.0, 0.0, -0.1),
  # Legged gait clock reward (not in Table II); added after ftsr_pretrain_rw_v1/v2
  # converged to standing still.
  "feet_gait": (0.0, 0.0, 1.5),
  # Not in Table II (released code's pen_torque_limits); off in the faithful baseline.
  "torque_limit": (0.0, 0.0, 0.0),
  # Soft speed caps (SPEED_CAPS); not in the paper, off in the faithful task.
  "joint_vel_excess": (0.0, 0.0, 0.0),
  "base_vz_excess": (0.0, 0.0, 0.0),
  "base_wxy_excess": (0.0, 0.0, 0.0),
}

# Speed above which the *_excess terms penalize (squared excess). Chosen for a get-up
# that a hardware test can follow: the faithful policy peaks at ~17 rad/s joint
# speed, 1.4 m/s torso vertical speed and ~14 rad/s roll/pitch rate (< 1 s get-up).
SPEED_CAPS = {"joint_vel": 4.0, "base_vz": 0.3, "base_wxy": 1.5}
# The caps' weights ramp in linearly over this many iterations after (re)start.
SPEED_CAP_RAMP_ITERATIONS = 300
