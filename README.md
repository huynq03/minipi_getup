# minipi_getup

Get-up (supine → standing) task for the HighTorque **Mini-Pi** 12-DOF biped, built on
[mjlab](https://github.com/mujocolab/mjlab).

Extracted from the Mini-Pi task in `mjlab_playground` (the generic getup MDP is adapted
from MuJoCo Playground). Only the Mini-Pi task is included.

## Task

| Task ID | Description |
|---------|-------------|
| `Mjlab-Getup-Flat-MiniPi` | Stand up from one fixed supine pose on flat ground |

- Reset: lying on the back (root z 0.085 m, pitched -88.3° about y), all joints 0 rad
  ± 0.025 rad noise, zero velocity. Actions are held for 0.2 s after reset.
- Action: 12-D relative joint-position targets (scale 0.6), PD actuators capped at
  ±16 Nm.
- Slow getup: penalties stay small while the getup is discovered, then ramp up.
  action_rate_l2 goes -0.01 → -0.03 → -0.05 (iterations 400, 700) and joint_vel_l2
  goes 0 → -0.002 → -0.005 → -0.01 (400, 700, 1000). The energy termination stays at
  an inf threshold (no measured Mini-Pi power limits).
- No spinning or shuffling: base yaw rate is penalized (-0.01 → -0.05 → -0.1 at 400,
  700), and so is horizontal base velocity once upright.
- Wide stance allowed: hip-roll posture std is 0.5 rad, and opening the hip roll toward
  its outward limit is rewarded while the base is below 0.2 m.
- Simulation: 2 ms physics step, 50 Hz policy, 10 s episodes.
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
