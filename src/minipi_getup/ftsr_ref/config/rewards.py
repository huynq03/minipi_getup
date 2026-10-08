"""Stage-wise reward table: getup_gym release groups mapped to Mini-Pi.

Release groups (``bipedal_wheeled/config.py``): ``reward_group_2/3/4`` are the three
groups the release actually selects (g1 is never used, quirk Q5) and play the roles
of the paper's r_u / r_s / r_w. Weights are the release's, unchanged. Only
robot-specific geometry is rescaled; dimensioned coefficients on torques,
accelerations, power and action rates are NOT rescaled (the archived TORQUE_SCALE = 36
is dropped).

| Release term | Release formula | g2 | g3 | g4 | Paper Table II | Mini-Pi form | Reason |
|---|---|---|---|---|---|---|---|
| pen_termination | reset & ~time_out | -200 | -100 | -500 | Termination | same | release |
| pen_base_orientation_l2 | 0.02|g_xy|^2+2(g_z+1)^2-clamp(e^{-|g_xy|^2/0.25},0,0.1) | -2.6 | -2.0 | -15 | Orient. | same | dimensionless |
| pen_base_orientation_z_l2 | 1[-g_z<-0.2] | -0.6 | -1.0 | - | (not listed) | same | dimensionless |
| track_base_height_exp | exp(-|h-h_cmd|/0.12) | 4 | 6 | 5 | Base heig. exp(-8.3 dh^2) | sigma 0.12 L/L_ref = 0.0552 m | ROBOT_ADAPTATION (length) |
| pen_dof_acc_l2 | sum(((qd_{t-1}-qd_t)/0.02)^2) | -2.5e-8 | -2.5e-6 | -2.5e-6 | DOF acc. | same, 12 joints | release |
| pen_action_rate_l2 | sum((a_{t-1}-a_t)^2) | -0.06 | -0.02 | -0.1 | Action rate | same | release |
| pen_dof_pos_bias_l2 | (sum_j[(qL-d)^2+|qR-d|])^2 * 1[2 contacts] | -0.06 | -0.14 | -0.12 | Dof pos. | 6 joints/leg, feet | ROBOT_ADAPTATION (joints) |
| pen_two_leg_bias_l2 | sum_j (qL-qR)^2 | -0.02 | -0.02 | - | Leg bias | 6 joint pairs | ROBOT_ADAPTATION (joints) |
| pen_no_fly_l2 | 1[#wheels F_z>1N != 2] | -2.0 | - | -0.2 | No fly | feet, 1 N x m/m_ref | ROBOT_ADAPTATION (mass) |
| rew_wheel_contact_force | exp(-0.05|sum F_z - 10 M|) | 4.0 | - | - | Wheel force | feet, width 20 N x m/m_ref | ROBOT_ADAPTATION (wheels->feet, mass) |
| pen_torques_l2 | sum tau^2 | - | -1e-6 | -1e-5 | Torques | same | release |
| track_lin_vel_xy_exp | exp(-|v-v_cmd|^2/0.12)*1[h>0.6] | - | - | 7 | Lin. vel. | sigma 0.12 L/L_ref, gate 0.6 L/L_ref | ROBOT_ADAPTATION (Froude) |
| track_ang_vel_yaw_exp | exp(-(w-w_cmd)^2/0.12) | - | - | 4 | Ang. vel. | sigma 0.12 L_ref/L | ROBOT_ADAPTATION (Froude) |
| pen_dof_vel_l2 | sum qd^2 (legs) | - | - | -4e-3 | Dof vel. | 12 joints | release |
| pen_feet_distance_l2 | clip(0.32-d,0,1)+clip(d-0.38,0,1) | - | - | -2.0 | Feet dist. | band (0.32, 0.38)/0.35 x d0 | ROBOT_ADAPTATION (geometry) |
| pen_max_velocity_l2 | +1 in [0.8,1.1] cmd else -1 | - | - | +1.0 | (not listed) | same | release |
| pen_dof_pos_limits | violation beyond 0.9 range | - | - | -5.0 | (not listed) | XML ranges | release |
| pen_torque_limits | sum relu(|tau|-0.8 tau_lim) | - | - | -0.05 | (not listed) | tau_lim = 16 Nm cap | ROBOT_ADAPTATION (motor) |
| pen_action_smoothness_l2 | sum(a_t-2a_{t-1}+a_{t-2})^2 | - | - | -0.02 | Action smooth. | same | release |
| pen_joint_power_l2 | sum|qd||tau| * 1[h<0.7] | - | - | -1e-4 | Dof ener. | gate 0.7 L/L_ref | ROBOT_ADAPTATION (length) |
| qd_soft_envelope | (not in release) | | | | | sum relu(|qd|-6.28)^2, -0.01 all stages (pd16_noslew; v0-v2: 3.0) | HARDWARE_CONSTRAINT (user operational contract) |

L = 0.345 m (Mini-Pi stance), L_ref = 0.75 m; m = 6.94 kg, m_ref = 27.669 kg.
"""

from minipi_getup.ftsr_ref.config.robot import (
  GRAVITY,
  LENGTH_RATIO,
  MINIPI_MASS,
  MINIPI_WEIGHT,
  REF_MASS,
  TAU_CAP,
)

FEET_SENSOR = "feet_contact"
MASS_RATIO = MINIPI_MASS / REF_MASS
CONTACT_THRESHOLD = 1.0 * MASS_RATIO  # N (release 1 N)
# Lateral xy distance of the ankle_roll_link origins in the nominal stance (measured,
# validate.py) and the release's band around its nominal 0.35 m.
FEET_DISTANCE_NOMINAL = 0.16
FEET_BAND = (0.32 / 0.35 * FEET_DISTANCE_NOMINAL, 0.38 / 0.35 * FEET_DISTANCE_NOMINAL)

# name: ((w_u, w_s, w_w), params)
REWARD_TABLE: dict[str, tuple[tuple[float, float, float], dict]] = {
  "pen_termination": ((-200.0, -100.0, -500.0), {}),
  "pen_base_orientation_l2": ((-2.6, -2.0, -15.0), {}),
  "pen_base_orientation_z_l2": ((-0.6, -1.0, 0.0), {}),
  "track_base_height_exp": ((4.0, 6.0, 5.0), {"sigma": 0.12 * LENGTH_RATIO}),
  "pen_dof_acc_l2": ((-2.5e-8, -2.5e-6, -2.5e-6), {}),
  "pen_action_rate_l2": ((-0.06, -0.02, -0.1), {}),
  "pen_dof_pos_bias_l2": (
    (-0.06, -0.14, -0.12),
    {"sensor_name": FEET_SENSOR, "contact_threshold": CONTACT_THRESHOLD},
  ),
  "pen_two_leg_bias_l2": ((-0.02, -0.02, 0.0), {}),
  "pen_no_fly_l2": (
    (-2.0, 0.0, -0.2),
    {"sensor_name": FEET_SENSOR, "contact_threshold": CONTACT_THRESHOLD},
  ),
  "rew_wheel_contact_force": (
    (4.0, 0.0, 0.0),
    {
      "sensor_name": FEET_SENSOR,
      "weight": MINIPI_WEIGHT,
      "width": 20.0 * MASS_RATIO * GRAVITY / 9.81,
    },
  ),
  "pen_torques_l2": ((0.0, -1.0e-6, -1.0e-5), {}),
  "track_lin_vel_xy_exp": (
    (0.0, 0.0, 7.0),
    {"sigma": 0.12 * LENGTH_RATIO, "min_height": 0.6 * LENGTH_RATIO},
  ),
  "track_ang_vel_yaw_exp": ((0.0, 0.0, 4.0), {"sigma": 0.12 / LENGTH_RATIO}),
  "pen_dof_vel_l2": ((0.0, 0.0, -4.0e-3), {}),
  "pen_feet_distance_l2": ((0.0, 0.0, -2.0), {"lo": FEET_BAND[0], "hi": FEET_BAND[1]}),
  "pen_max_velocity_l2": ((0.0, 0.0, 1.0), {}),
  "pen_dof_pos_limits": ((0.0, 0.0, -5.0), {"soft": 0.9}),
  "pen_torque_limits": (
    (0.0, 0.0, -0.05),
    {"limit": TAU_CAP, "soft": 0.8},
  ),
  "pen_action_smoothness_l2": ((0.0, 0.0, -0.02), {}),
  "pen_joint_power_l2": ((0.0, 0.0, -1.0e-4), {"max_height": 0.7 * LENGTH_RATIO}),
  # pd16_noslew: limit 6.28 rad/s (was 3.0); same weight, all stages.
  "qd_soft_envelope": ((-0.01, -0.01, -0.01), {"limit": 6.28}),
}
