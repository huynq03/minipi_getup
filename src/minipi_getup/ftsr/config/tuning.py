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
