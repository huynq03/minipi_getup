#!/usr/bin/env bash
# Play a HoST staged-recovery checkpoint in the viser viewer (pull force, DR and training
# terminations off), then open http://localhost:<port>.
#
#   scripts/play.sh                                   # checkpoints/rev13/model_1100.pt
#   scripts/play.sh path/to/model_N.pt -n 16 -p 8081
#
# Options: -n robots (1)  -s action scale (default: the checkpoint's training value)
#          -e episode seconds (10)  -p port (8080)  -d domain randomization on
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ckpt="$ROOT/checkpoints/rev13/model_1100.pt" envs=1 scale="" episode=10 port=8080 dr=off
if [[ $# -gt 0 && "$1" != -* ]]; then ckpt="$1"; shift; fi
while getopts "n:s:e:p:dh" opt; do
  case "$opt" in
    n) envs="$OPTARG" ;; s) scale="$OPTARG" ;; e) episode="$OPTARG" ;; p) port="$OPTARG" ;;
    d) dr=on ;; h) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; *) exit 1 ;;
  esac
done
[[ -f "$ckpt" ]] || { echo "checkpoint not found: $ckpt" >&2; exit 1; }

cd "$ROOT"
if [[ -d /run/opengl-driver/lib ]]; then  # NixOS: driver and libglvnd are off the loader path
  glvnd=$(ls -d /nix/store/*-libglvnd-*/lib 2>/dev/null | head -1 || true)
  export LD_LIBRARY_PATH="/run/opengl-driver/lib${glvnd:+:$glvnd}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT/.warp-cache-cu12}"
PY=.venv/bin/python
if [[ -z "$scale" ]]; then
  scale=$($PY -c "import sys, torch
from minipi_getup.host_staged_recovery.export_onnx import checkpoint_action_scale
s = checkpoint_action_scale(torch.load(sys.argv[1], map_location='cpu', weights_only=False))
print(f'{s:.3f}' if s is not None else '')" "$ckpt" 2>/dev/null | tail -1)
  [[ -n "$scale" ]] || { scale=0.5; echo "no action scale in the checkpoint; using $scale" >&2; }
fi
echo "checkpoint: $ckpt"
echo "viewer:     http://localhost:$port  (robots $envs, action scale $scale, DR $dr)"
exec $PY -m minipi_getup.host_staged_recovery.play_viser --checkpoint "$ckpt" --num-envs "$envs" \
  --action-scale "$scale" --episode-s "$episode" --dr "$dr" --port "$port"
