# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An mjlab task package for the get-up task (supine → standing) on the HighTorque Mini-Pi 12-DOF biped.
It registers a single task, `Mjlab-Getup-Flat-MiniPi`, and depends on released `mjlab[cu128]==1.6.0` from PyPI.
The parent `/home/hai/huy/CLAUDE.md` describes the sibling `minipi_velocity` package. Its commands and paths
(for example `~/.local/bin/uv` and `/home/huynq/mjlab`) don't apply on this machine.

## Machine setup (NixOS, GTX 1080 Ti)

- `uv` is on PATH. Run `uv sync` to create `.venv`.
- Every `uv run train/play/list-envs` or Python script that builds the env needs the driver and glvnd libraries
  on the loader path (fish syntax):
  ```sh
  set -x LD_LIBRARY_PATH /run/opengl-driver/lib:/nix/store/jvq8nlprvmwi0qpk46iay198qzcav962-libglvnd-1.7.0/lib
  ```
  - Without `/run/opengl-driver/lib`, torch and Warp don't see CUDA.
  - Without libglvnd, `import mjlab` fails in PyOpenGL EGL with `'NoneType' object has no attribute
    'eglQueryString'`, because mjlab defaults `MUJOCO_GL=egl`.
  - The glvnd store path comes from the system closure: re-find it with
    `nix-store -qR /run/current-system | grep libglvnd` after a rebuild.
- The GPU is Pascal (sm_61). Two workarounds are in place, and both are needed:
  - `pyproject.toml` pulls torch from the **cu126** index. The cu128 wheels have no sm_61 kernels.
  - `src/minipi_getup/__init__.py` disables Warp's libmathdx (`enable_mathdx_solver/gemm = False`) when any GPU
    is below sm_70. cuSolverDx rejects sm_61, and mujoco_warp's `tile_cholesky` (mass matrix and Newton
    solver) would otherwise fail to compile. This must run before any Warp kernel compiles. The task entry
    point is imported by `import mjlab`, so that holds.

## Commands

```sh
uv run list-envs
uv run train Mjlab-Getup-Flat-MiniPi --env.scene.num-envs 4096 --agent.logger tensorboard [--agent.run-name NAME]
uv run play Mjlab-Getup-Flat-MiniPi --checkpoint-file logs/rsl_rl/minipi_getup/<run>/model_<N>.pt --num-envs 16 --viewer viser
.venv/bin/ruff check src && .venv/bin/ruff format --check src     # 2-space indent
```

- mjlab's default logger is W&B, which isn't logged in here. Pass `--agent.logger tensorboard`, or the run
  stalls at a login prompt.
- Long training runs go in the `getup` tmux session. Leave the user's `minipi` session alone.
- Checkpoints go to `logs/rsl_rl/minipi_getup/<timestamp>[_run-name]/model_<iter>.pt` every 50 iterations.
  Nothing tracks a "best" checkpoint. To choose one, read the TensorBoard event file there
  (`Train/mean_reward`, `Episode_Metrics/getup_success`).
- There are no tests. To check a change:
  - Run a short smoke train (`--agent.max-iterations 3`).
  - Or build `ManagerBasedRlEnv(load_env_cfg("Mjlab-Getup-Flat-MiniPi"), device=...)` in a script and step it.
  - For behavior questions (spinning, stance), roll out a checkpoint with `MjlabOnPolicyRunner.load(...)` +
    `get_inference_policy()` and measure root state, the way `mjlab/scripts/play.py` does.

## Architecture

- **Discovery:** the `[project.entry-points."mjlab.tasks"]` entry makes `import mjlab` import `minipi_getup`.
  `getup/config/minipi/__init__.py` calls `register_mjlab_task` with the train and play env cfgs plus the PPO
  cfg from `rl_cfg.py`.
- **Two-layer config:**
  - `getup/getup_env_cfg.py::make_getup_env_cfg()` is a robot-agnostic get-up env adapted from MuJoCo
    Playground. Its defaults are a random fallen-or-standing reset, a 5 ms timestep with decimation 4, 6 s
    episodes, and a T1-style curriculum.
  - `getup/config/minipi/env_cfgs.py::minipi_getup_env_cfg()` overrides most of that for Mini-Pi:
    - 2 ms timestep with decimation 10 (50 Hz).
    - A fixed supine reset (`reset_fixed_pose`) in place of the random one.
    - Its own episode length, reward terms, posture stds and curriculum.
  - The Mini-Pi file is where task tuning happens.
- **`getup/mdp/`** re-exports `mjlab.envs.mdp` and adds get-up terms:
  - `SettleRelativeJointPositionAction`: holds actions for `settle_steps` after reset. The reset events set
    `env.extras["settle_mask"]`.
  - Rewards: `orientation_reward`, `height_reward`, `gated_posture_reward` and the `getup_success` metric. The
    Mini-Pi cfg also uses `base_yaw_rate_l2`, `upright_base_lin_vel_xy_l2` and `hip_roll_open_when_low`.
  - `energy_termination`.
- **Reward logging:** each `Episode_Reward/<term>` is the episode sum divided by `episode_length_s` (a
  per-second average), and each step's reward is scaled by `step_dt`. `Mean reward` is the mean episode return.
- **Curriculum stage `step`s** are env steps, not iterations. Write them as `<iteration> * 24`, since
  `num_steps_per_env=24`.
- **Robot:**
  - `asset_zoo/robots/hightorque_minipi/minipi_constants.py` loads `xmls/cl_pai.xml`.
  - PD actuators are capped at 16 Nm.
  - Zero joint angles are the standing pose; the knee bend is baked into the body frames.
  - `soft_joint_pos_limit_factor` is 1.0, so `dof_pos_limits` only penalizes going past the XML ranges.
  - Hip-roll ranges are asymmetric (right `-0.5..0.12`, left `-0.12..0.5`), and the larger side is outward.
  - The target torso height (`_TORSO_HEIGHT = 0.344`) and the supine root pose were measured in sim. Re-measure
    them if the reset joint pose changes.
