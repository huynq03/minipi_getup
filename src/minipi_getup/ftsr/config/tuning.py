"""FTSR tuning variants: stage-weight overrides on top of the faithful task.

Each variant is one hypothesis, one parameter family (docs/FTSR_EXPERIMENT_LOG.md).
The faithful reproduction ``Mjlab-FTSR-MiniPi`` is never changed by these.
"""

from minipi_getup.ftsr.config.env_cfg import ACTION_SCALE
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

# Get-up-first variant (user request, 2026-10-07): a get-up slow and gentle enough to
# try on hardware, with the robot only standing afterwards (``stand_only``). In r_w
# every second spent upright earns the height and zero-command tracking rewards, so
# rising faster always pays, and nothing penalizes speed. Soft caps on joint speed,
# torso vertical speed and roll/pitch rate (SPEED_CAPS, squared excess, ramped in)
# act in every stage. Rough balance: the faithful get-up's peaks (17 rad/s over
# ~3 joints, 1.4 m/s, 14 rad/s, each for ~0.2 s) would cost ~40, 11 and 29 per
# episode, against ~16 gained by standing 1 s earlier.
GETUP_SOFT = {
  "joint_vel_excess": (-0.5, -0.5, -0.5),
  "base_vz_excess": (-50.0, -50.0, -50.0),
  "base_wxy_excess": (-1.0, -1.0, -1.0),
}

# Get-up-first round 2. GetupSoft v1 (speed caps only) slowed the get-up from 0.73 s
# to ~1.3 s, but peak joint speed stayed ~16 rad/s: the incentive to rise early (height
# and tracking rewards for every second upright) still outweighed the caps. v2 removes
# that incentive at its source with a rise schedule (stance height reached 2.5 s after
# the settle hold; height above schedule + 3 cm is penalized) and triples the joint
# speed cap weight. Fine-tunes the v1 policy.
GETUP_SOFT_V2 = {
  **GETUP_SOFT,
  "joint_vel_excess": (-1.5, -1.5, -1.5),
  "height_schedule": (-100.0, -100.0, -100.0),
}

# Get-up-first round 3: hard limits instead of reward shaping. Rounds 1-2 showed that
# speed penalties on a policy that already jumps up only stretch its slow phases, and
# collapse it once they bind. Here the joint target may move at most
# GENTLE_TARGET_STEP rad per 20 ms (5 rad/s), so a jump can't be commanded at all,
# and the torque envelope is raised to GENTLE_TORQUE_LIMIT (user, 2026-10-07: 12-13
# Nm) so that a slow get-up has the strength it needs. Trained from the walking
# pretrain with the assist, like the faithful run, since the old policy relies on
# fast targets. Rewards: faithful stage table, zero commands.
GENTLE_TORQUE_LIMIT = 12.5
GENTLE_TARGET_STEP = 0.1

# Get-up-first round 4 (deployable): the env matches what mini_pi_fsm can run with a
# config-only policy package. (a) HTDW-5036 torque-speed curve (21 Nm stall, 7.85
# rad/s no-load) with the motors' real 16 Nm cap, since deployment doesn't clamp
# torque; the user's 12.5 Nm envelope becomes a penalty (|tau| above it, -1/s per Nm).
# (b) The rate limit moves from the joint target to the action, measured from
# last_action and starting at 0, so the exported ONNX can apply the very same clamp:
# 0.4 action units = 0.1 rad of target per 20 ms step. Fine-tunes GetupGentle v1.
DEPLOY_TORQUE_ENVELOPE = 12.5
DEPLOY_MOTOR_CAP = 16.0
DEPLOY_ACTION_STEP = GENTLE_TARGET_STEP / ACTION_SCALE
GETUP_DEPLOY = {"torque_limit": (-1.0, -1.0, -1.0)}

# Round 4, v3: without the assist the fine-tune stalled (it 3500 -> 3882: ~40 %
# standing, prone not relearned). Re-run the FTSR curriculum on the deploy env from the
# Gentle weights, with a short assist schedule (off at it 1000 of the new run).
DEPLOY_ASSIST_END_ITERATION = 1000
# v3 collapsed at it ~420 (population above h1 fell to 0): the stage manager isn't
# checkpointed, so every fine-tune restarts in r_u, and with the new motor model only
# ~50 % stood, below the 2/3 needed to leave it; r_u (height target 0.19 m) pulled the
# policy away from standing. v2 stalled in r_u too. v4 fixes the stage at r_w, the
# reward the Gentle policy was trained under.
DEPLOY_FIXED_STAGE = 2
