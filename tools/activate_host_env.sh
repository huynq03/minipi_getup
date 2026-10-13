#!/usr/bin/env bash
# Source this file from Bash: source tools/activate_host_env.sh
# Activates dependencies and local caches only; never starts training.
HOST_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$HOST_ENV_ROOT/.venv/bin/activate" ]]; then
  echo "Missing .venv; run uv sync --locked first." >&2
  return 1 2>/dev/null || exit 1
fi
export UV_CACHE_DIR="$HOST_ENV_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$HOST_ENV_ROOT/.uv-python"
export XDG_CACHE_HOME="$HOST_ENV_ROOT/.cache"
export MPLCONFIGDIR="$HOST_ENV_ROOT/.cache/matplotlib"
export CUDA_CACHE_PATH="$HOST_ENV_ROOT/.cache/cuda"
export WARP_CACHE_PATH="$HOST_ENV_ROOT/.warp-cache-cu12"
export MUJOCO_GL=egl
source "$HOST_ENV_ROOT/.venv/bin/activate"
unset HOST_ENV_ROOT
