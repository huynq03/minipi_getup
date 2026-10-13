# Training the staged-recovery policy on another machine

This explains how to clone the `host-only` branch and train the HoST Mini-Pi staged-recovery policy
(revision 13) on any Linux machine with an NVIDIA GPU. The branch contains everything a run needs:
the code, the robot asset (XML and meshes), `uv.lock` and the launch scripts. Only checkpoints
(`*.pt`) and logs are not tracked.

## 1. Clone and install

```bash
git clone -b host-only git@github.com:huynq03/minipi_getup.git minipi_getup_host
cd minipi_getup_host
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is missing
uv sync                                            # Python 3.13, torch (cu126), mjlab 1.6.0, warp 1.18 (cu12)
```

**Requirements:**

- An NVIDIA driver with CUDA 12 support (≥ 550).
- About 10 GB of GPU memory for 4096 environments. Lower `-n` on smaller GPUs.

**Pascal GPUs (sm_61, e.g. GTX 1080 Ti) are handled automatically:**

- torch comes from the cu126 index.
- `minipi_getup/__init__.py` disables Warp's libmathdx.

**NixOS:** the scripts add `/run/opengl-driver/lib` and the libglvnd store path by themselves. Other
distributions need nothing extra.

**Check the install:**

```bash
.venv/bin/python -m unittest tests.test_host_staged_recovery tests.test_host_safe_recovery   # 62 tests
```

## 2. Train

Run inside tmux so the run survives a closed terminal:

```bash
tmux new -s host_train
tools/train_host_staged.sh                 # from scratch: 4096 envs, 8000 iterations, run name rev13
```

The script has these options:

| Option | Meaning | Default |
|---|---|---|
| `-n N` | number of parallel environments | 4096 |
| `-i N` | final iteration (total) | 8000 |
| `-r FILE` | resume from a checkpoint `model_<N>.pt`: policy, optimizer and curriculum state | – |
| `-N NAME` | run-name suffix | `rev13` |
| `-s N` | seed | 1 |
| `-b` | start even if the GPU already has processes (e.g. a viewer) | off |

**What gets written, and how long it takes:**

- Output goes to `logs/host_staged_recovery/<YYYYmmdd_HHMMSS>_<name>/`. It contains `model_<iter>.pt`
  every 100 iterations, TensorBoard events, and `config.json` (commit, asset hashes, reward scales).
- The console log also goes to `logs/host_staged_recovery/train_<stamp>_<name>.log`.
- On an RTX 3090, 4096 envs run at about 3.9 s per iteration, so 8000 iterations take about 8.7 h.

**Training refuses to start in two cases:**

- the package has uncommitted changes (commit first: the run records its commit);
- the GPU is busy (use `-b`).

### Resuming a checkpoint from another machine

Checkpoints are not in git. Copy one over, then resume:

```bash
scp user@host:/path/to/logs/host_staged_recovery/<run>/model_1100.pt ckpt/
tools/train_host_staged.sh -r ckpt/model_1100.pt -i 8000 -N rev13_from1100
```

A checkpoint resumes only with the same variant and the same robot asset hash. The tracked XML is
the one every revision-13 checkpoint was trained with, so this holds.

## 3. Monitor

```bash
.venv/bin/tensorboard --logdir logs/host_staged_recovery --port 6006
```

Useful scalars (`Episode_end/…`):

| Area | Scalars |
|---|---|
| Getting up | `fallen_success`, `rise_two_foot_frac`, `rise_one_foot_frac` |
| Curriculum | `penalty_foot_success_ema` (starts at 0.60, full at 0.85), `Episode/action_scale` (floor 0.30) |
| Standing | `standing_min_sole_coverage`, `standing_both_soles_cov90_frac`, `standing_torso_lateral_err_m`, `standing_com_margin_m` |
| Stepping | `standing_lifts_per_ep`, `standing_stance_offset_err_m`, `post_stand_drag_m`, `stance_start_success`, `stance_start_lifts_per_ep`, `step_placement_gain` |

## 4. Watch a checkpoint

```bash
tools/play_host_staged.sh                 # latest checkpoint of the latest run, 1 robot, training action scale
tools/play_host_staged.sh 1100 rev13      # model_1100 of the latest run whose name contains "rev13"
tools/play_host_staged.sh -l              # list runs and their latest checkpoints
```

Then open `http://localhost:8080` (choose another port with `-p`). Playback disables the pull force,
domain randomization, the training terminations and the stance starts.

## Reward design

- `src/minipi_getup/host_staged_recovery/REWARDS.md`: every term, with its equation and revision
  history.
- `reports/host/REV13_REWARD_CHANGES.md`: the revision-13 changes, grouped by HoST reward group, and
  the run history.
