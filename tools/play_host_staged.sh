#!/usr/bin/env bash
# Play a recovery checkpoint in the viser viewer (pull force, DR and velocity
# termination off), then open http://localhost:<port>.
#
#   tools/play_host_staged.sh                 # latest checkpoint of the latest staged run
#   tools/play_host_staged.sh 600             # model_600 of the latest staged run
#   tools/play_host_staged.sh 600 rev4        # model_600 of the latest run whose name contains "rev4"
#   tools/play_host_staged.sh path/to/model_600.pt
#   tools/play_host_staged.sh -l              # list runs and their checkpoints
#
# Options (anywhere on the command line):
#   -v staged|safe   experiment (default staged)
#   -n N             number of robots (default 1)
#   -s SCALE         action scale (default: the training action scale logged at
#                    the checkpoint's iteration, Episode/action_scale)
#   -e SECONDS       episode length (default 10)
#   -p PORT          viewer port (default 8080)
#   -d               domain randomization on
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant=staged envs=1 scale=auto episode=10 port=8080 dr=off list=0
positional=()
while [[ $# -gt 0 ]]; do  # options may come before or after the positional arguments
  case "$1" in
    -v) variant="$2"; shift ;;
    -n) envs="$2"; shift ;;
    -s) scale="$2"; shift ;;
    -e) episode="$2"; shift ;;
    -p) port="$2"; shift ;;
    -d) dr=on ;;
    -l) list=1 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1 (-h for help)" >&2; exit 1 ;;
    *) positional+=("$1") ;;
  esac
  shift
done
set -- "${positional[@]+"${positional[@]}"}"

case "$variant" in
  staged|safe) ;;
  *) echo "unknown variant: $variant (staged|safe)" >&2; exit 1 ;;
esac
LOGS="$ROOT/logs/host_${variant}_recovery"

latest_model() {  # highest-numbered model_*.pt in a run directory
  find "$1/" -maxdepth 1 -name 'model_*.pt' | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1
}

if [[ $list == 1 ]]; then
  for run in $(find "$LOGS" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | sort); do
    last=$(latest_model "$run")
    printf '%-50s latest model_%s\n' "$(basename "$run")" "${last:--}"
  done
  exit 0
fi

if [[ $# -ge 1 && "$1" == *.pt ]]; then
  ckpt="$1"
else
  iter="${1:-latest}"
  pattern="${2:-}"
  run=$(find "$LOGS" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) -name "*${pattern}*" | sort | tail -1)
  [[ -n "$run" ]] || { echo "no run matching '$pattern' in $LOGS" >&2; exit 1; }
  [[ "$iter" == latest ]] && iter=$(latest_model "$run")
  ckpt="$run/model_${iter}.pt"
fi
[[ -f "$ckpt" ]] || { echo "checkpoint not found: $ckpt" >&2; exit 1; }
if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
  echo "port $port is already in use (another viewer?); pick another with -p" >&2; exit 1
fi

if [[ "$scale" == auto ]]; then
  # Training action scale at this checkpoint: last logged value at or before its iteration.
  scale=$("$ROOT/.venv/bin/python" - "$ckpt" <<'PY' 2>/dev/null || true
import glob, os, re, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ckpt = sys.argv[1]
it = int(re.search(r"model_(\d+)\.pt$", ckpt).group(1))
values = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(ckpt), "events.out.tfevents.*"))):
  ea = EventAccumulator(f, size_guidance={"scalars": 0})
  ea.Reload()
  if "Episode/action_scale" in ea.Tags()["scalars"]:
    values += [(e.step, e.value) for e in ea.Scalars("Episode/action_scale")]
past = [v for s, v in values if s <= it]
if past:
  print(f"{past[-1]:.3f}")
PY
)
  if [[ -z "$scale" ]]; then
    scale=0.5
    echo "no logged action scale for this checkpoint; using $scale" >&2
  else
    scale_note=" (training value at this checkpoint)"
  fi
fi

echo "checkpoint: $ckpt"
echo "viewer:     http://localhost:$port  (envs $envs, action scale $scale${scale_note:-}, DR $dr)"
cd "$ROOT"
# NixOS: the driver libraries are not on the default loader path.
if [[ -d /run/opengl-driver/lib ]]; then
  glvnd=$(ls -d /nix/store/*-libglvnd-*/lib 2>/dev/null | head -1 || true)
  export LD_LIBRARY_PATH="/run/opengl-driver/lib${glvnd:+:$glvnd}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT/.warp-cache-cu12}"
exec .venv/bin/python -m "minipi_getup.host_${variant}_recovery.play_viser" \
  --checkpoint "$ckpt" --num-envs "$envs" --action-scale "$scale" \
  --episode-s "$episode" --dr "$dr" --port "$port"
