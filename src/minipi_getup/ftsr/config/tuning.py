"""FTSR tuning variants: stage-weight overrides on top of the faithful task.

Each variant is one hypothesis, one parameter family (docs/FTSR_EXPERIMENT_LOG.md).
The faithful reproduction ``Mjlab-FTSR-MiniPi`` is never changed by these.
"""

from minipi_getup.ftsr.config.stage_rewards import STAGE_WEIGHTS

# Round 1, recovery-phase motion regularization. The faithful policy got up in ~1 s
# with peak joint speeds of 17 rad/s and roll/pitch rates of 13 rad/s, as violent as
# the old baseline, because r_u and r_s don't penalize speed at all. An L2 joint
# velocity cost always favors slower motion (moving an angle d in time T costs
# ~d^2/T), so it can't make the get-up infeasible the way a hard limit can. The
# released code's soft torque-limit term (|tau| above 0.8 x 9 Nm) is switched on in
# every stage.
SMOOTH_V1 = {
  "dof_vel": (-0.05, -0.05, STAGE_WEIGHTS["dof_vel"][2]),
  "torque_limit": (-2.0, -2.0, -2.0),
}

# Round 2, motion regularization in the stage that actually shapes the get-up. From
# it ~815 the population is in r_w, so every get-up afterwards is optimized under the
# walking weights. Round 1 changed r_u/r_s (no effect once in r_w) and, applied from
# scratch, stopped the get-up from ever being discovered. Round 2 fine-tunes the
# faithful policy (resumed at it 4000, assist already off) with a stronger r_w joint
# speed cost (walking runs at ~1-2 rad/s, the get-up peaks at 17 rad/s) and the soft
# torque-limit term (walking torque p99 was ~8 Nm against the 9 Nm cap).
SMOOTH_V2 = {
  "dof_vel": (*STAGE_WEIGHTS["dof_vel"][:2], -0.04),
  "torque_limit": (0.0, 0.0, -1.0),
}
