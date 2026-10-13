# Local training environment — air3070

Setup verified on 2026-10-13 for source commit `2852bd5` and revision 13.
Repository subsequently updated to `8636ae6` to retrieve the tracked resume checkpoint.
No training runner, policy optimization, or checkpoint creation was executed.

## Activate from the repository root

```bash
source tools/activate_host_env.sh
```

This activates `.venv` and sets local uv, Matplotlib, CUDA and Warp caches,
plus `MUJOCO_GL=egl`. Managed Python lives in `.uv-python`; keep that directory
because `.venv` uses it. System Python and NVIDIA drivers were not modified.

Dependencies were installed with `uv sync --locked`; `uv.lock` is unchanged.
After activation, `uv sync --locked --offline` also succeeds.

| Component | Verified version |
|---|---|
| OS | Ubuntu 22.04, Linux 6.8.0-138-generic |
| GPU | RTX 3070 Ti, 8 GiB, sm_86 |
| NVIDIA driver | 580.178.04 |
| Python | 3.13.16 |
| PyTorch | 2.9.1+cu126, CUDA runtime 12.6 |
| mjlab | 1.6.0 |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0 |
| Warp | 1.18.0+cu12 |
| TensorBoard | 2.21.0 |

## Verification completed

- `uv pip check`: all 136 installed packages compatible.
- Existing unit tests: 62 passed.
- Asset manifest: XML checksum verified, all 14 referenced files readable and hashed.
- Rev13 settings: drag termination off, action-scale floor 0.30.
- PyTorch and Warp both detect the NVIDIA GPU.
- Two CUDA environments reset and step three times with zero actions; observations
  and rewards remain finite. Warp kernels compiled and cached.
- Training CLI `--help`, launcher shell syntax and setup-script lint checks passed.

Repeat the setup check without training:

```bash
source tools/activate_host_env.sh
python tools/check_host_env.py
```

GPU checks must run in a normal terminal with GPU access. The restricted Codex
sandbox hides `/dev/nvidia*`; checks here used approved access outside that sandbox.

## Before a future training run

The repository guide estimates about 10 GB VRAM for 4096 environments. This GPU
has 8 GB: start with 1024 environments and measure memory before increasing.
The setup check validates only two environments, not training memory capacity.
The launcher checks for active GPU compute processes by default; wait for those
to finish before starting a run.

The resume checkpoint is now available at `checkpoints/rev13/model_1100.pt`,
retrieved from commit `8636ae6` on `origin/host-only`. Its SHA256 is
`2a4451ae9a5293a74539c2c0377bc9633e6116a68e1ef84d782a804f247b3433`.
See `checkpoints/rev13/README.md` for its source run and resume command.
Fresh training does not require a checkpoint.

About 28 GB disk space remained after setup. Logs and future checkpoints need
additional space. See `docs/HOST_STAGED_TRAINING.md` for training and playback
commands; none of those training commands were run during this setup.
