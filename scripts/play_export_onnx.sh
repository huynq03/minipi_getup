#!/usr/bin/env bash
# Export a checkpoint's actor to ONNX (verified against PyTorch), then play it like scripts/play.sh.
# The ONNX file and its JSON metadata go to <checkpoint dir>/onnx/<checkpoint stem>.onnx.
#
#   scripts/play_export_onnx.sh                         # checkpoints/rev13/model_1100.pt
#   scripts/play_export_onnx.sh path/to/model_N.pt -n 16
#
# Options are those of scripts/play.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ckpt="$ROOT/checkpoints/rev13/model_1100.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then ckpt="$1"; shift; fi
[[ -f "$ckpt" ]] || { echo "checkpoint not found: $ckpt" >&2; exit 1; }
(cd "$ROOT" && .venv/bin/python -m minipi_getup.host_staged_recovery.export_onnx --checkpoint "$ckpt")
exec "$ROOT/scripts/play.sh" "$ckpt" "$@"
