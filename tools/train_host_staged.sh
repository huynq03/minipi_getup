#!/usr/bin/env bash
# Train the staged-recovery policy (HoST on Mini-Pi, revision 13) on any Linux +
# NVIDIA machine. Run it from a clone of the host-only branch, ideally in tmux.
#
#   tools/train_host_staged.sh                            # from scratch, 4096 envs, 8000 iterations
#   tools/train_host_staged.sh -n 2048 -i 3000            # fewer envs / iterations
#   tools/train_host_staged.sh -r path/to/model_1100.pt   # resume up to iteration -i (default 8000)
#   tools/train_host_staged.sh -N my_run                  # run name suffix (default rev13)
#   tools/train_host_staged.sh -b                         # start even if the GPU has other processes
#
# Checkpoints and TensorBoard go to logs/host_staged_recovery/<timestamp>_<name>/
# (every 100 iterations); the console log is also written next to them.
# On NixOS the CUDA driver and libglvnd paths are added automatically when
# found; elsewhere nothing extra is needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
envs=4096 total=8000 resume="" name=rev13 seed=1 busy=()
while getopts "n:i:r:N:s:bh" opt; do
  case "$opt" in
    n) envs="$OPTARG" ;;
    i) total="$OPTARG" ;;
    r) resume="$OPTARG" ;;
    N) name="$OPTARG" ;;
    s) seed="$OPTARG" ;;
    b) busy=(--allow-busy-gpu) ;;
    h) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) exit 1 ;;
  esac
done

cd "$ROOT"
if [[ -x .venv/bin/python ]]; then
  PY=(.venv/bin/python)
elif command -v uv >/dev/null; then
  uv sync
  PY=(.venv/bin/python)
else
  echo "no .venv and no uv: install uv (https://docs.astral.sh/uv/) and run 'uv sync'" >&2; exit 1
fi

# NixOS: the driver libraries are not on the default loader path.
if [[ -d /run/opengl-driver/lib ]]; then
  glvnd=$(ls -d /nix/store/*-libglvnd-*/lib 2>/dev/null | head -1 || true)
  export LD_LIBRARY_PATH="/run/opengl-driver/lib${glvnd:+:$glvnd}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT/.warp-cache-cu12}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

args=(--num-envs "$envs" --seed "$seed" --run-name "$name" "${busy[@]}")
if [[ -n "$resume" ]]; then
  [[ -f "$resume" ]] || { echo "checkpoint not found: $resume" >&2; exit 1; }
  start=$(sed -n 's/.*model_\([0-9]*\)\.pt$/\1/p' <<<"$resume")
  [[ -n "$start" ]] || { echo "cannot read the iteration from $resume (expected model_<N>.pt)" >&2; exit 1; }
  (( total > start )) || { echo "nothing to do: checkpoint $start >= $total" >&2; exit 1; }
  args+=(--resume "$resume" --max-iterations $((total - start)))
else
  args+=(--max-iterations "$total")
fi

mkdir -p logs/host_staged_recovery
log="logs/host_staged_recovery/train_$(date +%Y%m%d_%H%M%S)_${name}.log"
echo "commit $(git rev-parse --short HEAD)  ${resume:+resume $resume  }-> iteration $total, $envs envs; log $log"
"${PY[@]}" -m minipi_getup.host_staged_recovery.train "${args[@]}" 2>&1 | tee "$log"
