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
- Simulation: 2 ms physics step, 50 Hz policy, 6 s episodes.
- Observations: actor 42-D (base angular velocity, projected gravity, joint positions,
  joint velocities, last action); critic 45-D (adds base linear velocity).

## Getting started

```bash
uv sync
uv run train Mjlab-Getup-Flat-MiniPi --env.scene.num-envs 4096
uv run play Mjlab-Getup-Flat-MiniPi
```

## Dependencies

mjlab and mujoco_warp are pinned to the same commits as `mjlab_playground`
(mjlab 1.2.0 @ `f7fdb16`, mujoco_warp 3.6.0 @ `875c4ca`). MuJoCo is the stable PyPI
release 3.7.0 rather than the py.mujoco.org nightly that mjlab's own sources point to,
because those nightly wheels are no longer hosted.

## Layout

```text
src/minipi_getup/
  asset_zoo/robots/hightorque_minipi/   Mini-Pi MJCF, meshes and robot config
  getup/getup_env_cfg.py                Generic getup environment factory
  getup/mdp/                            Generic getup actions, events, rewards, terminations
  getup/config/minipi/                  Mini-Pi env config, PPO config, task registration
```
