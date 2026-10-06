# minipi_getup

Get-up (supine → standing) task for the HighTorque **Mini-Pi** 12-DOF biped, built on
[mjlab](https://github.com/mujocolab/mjlab).

Extracted from the Mini-Pi task in `mjlab_playground` (the generic getup MDP is adapted
from MuJoCo Playground). Only the Mini-Pi task is included.

## Task

| Task ID | Description |
|---------|-------------|
| `Mjlab-Getup-Flat-MiniPi` | Stand up from one fixed supine pose on flat ground |

- Reset: 60% lying on the back (root z 0.085 m, pitched -88.3° about y, all joints
  0 rad), 40% in one of three statically balanced crouches (base 0.19 / 0.23 / 0.27 m,
  torso upright, feet flat). All with ± 0.025 rad joint noise and zero velocity.
  Actions are held for 0.2 s after reset. The crouches exist because, within the
  hardware limits (hip flexion ≤ 1.25 rad), no balanced crouch has the base below
  ~0.19 m, so a sit on the ground (base ~0.06 m) cannot reach standing through balanced
  poses; crouched starts let the squat-to-stand part be learned.
- Action: 12-D relative joint-position targets (scale 0.25), clamped to the joint
  limits so no joint is driven into its hard stop. PD actuators capped at ±16 Nm.
- Slow getup: penalties stay small while the getup is discovered, then ramp up.
  action_rate_l2 goes -0.01 → -0.03 → -0.05 (iterations 400, 700) and joint_vel_l2
  goes 0 → -0.002 → -0.005 → -0.01 (400, 700, 1000). The energy termination stays at
  an inf threshold (no measured Mini-Pi power limits).
- Spinning is penalized from the start, including getup preparation: world-z angular
  velocity has L1 weight -0.5 and L2 weight -0.1. The L1 cost discourages slow full
  turns as well as fast ones. Horizontal base velocity is penalized once upright.
- Joint position limits match the robot's `clpai_12dof_0905` URDF, as provided by
  the user. Hip roll opens outward to 0.5 rad per side; there are no range overrides.
  Hip-roll posture std is 0.8 rad, and opening is rewarded below 0.2 m while the torso
  is lying (not sitting up and not inverted). Other posture stds are relaxed by 50%.
  PD stiffness is reduced by 30% with damping unchanged. Upward speed above 0.3 m/s is
  penalized while the base is below 0.12 m to discourage launching upward during
  preparation.
- Sit → squat → stand: sitting upright on the ground is not a resting point. Posture
  is rewarded only above 80% of standing height. Torso height has weight 2 and is
  scaled by uprightness, so lifting the base while lying or inverted earns nothing.
  Once the torso is upright-ish, the feet are rewarded for being under the hips
  (forward offset 0.055 m from base_link, std 0.08 m).
- Simulation: 2 ms physics step, 50 Hz policy, 20 s episodes (1000 steps).
- Observations: actor 42-D (base angular velocity, projected gravity, joint positions,
  joint velocities, last action); critic 45-D (adds base linear velocity).

## Getting started

```bash
uv sync
uv run train Mjlab-Getup-Flat-MiniPi --env.scene.num-envs 4096
uv run play Mjlab-Getup-Flat-MiniPi
```

## Dependencies

`mjlab[cu128]==1.6.0` from PyPI, which brings MuJoCo 3.11, MuJoCo Warp 3.11 and
RSL-RL 5.4.2. On Linux x86_64, `warp-lang` comes from NVIDIA's CUDA 12 release wheel
(`1.18.0+cu12`) because the PyPI wheels are CUDA 13 builds that need a 580+ driver.

## Layout

```text
src/minipi_getup/
  asset_zoo/robots/hightorque_minipi/   Mini-Pi MJCF, meshes and robot config
  getup/getup_env_cfg.py                Generic getup environment factory
  getup/mdp/                            Generic getup actions, events, rewards, terminations
  getup/config/minipi/                  Mini-Pi env config, PPO config, task registration
```
