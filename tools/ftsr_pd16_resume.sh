#!/usr/bin/env bash
# Experiment pd16_noslew: resume FTSR recovery from model_2700.pt after the
# non-finite constraint-cost stop (physics explosion fix). Restores networks, both
# optimizers, adaptive lr, iteration, env step counter (assist schedule) and RNG;
# continues to iteration 8000 in a new run dir, which inherits the best-checkpoint
# record of the original run. Run inside tmux from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
export WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12

SRC_NAME=2026-10-08_23-51-06_ftsr_recovery_pd16_noslew
SRC=logs/rsl_rl/minipi_ftsr_ref/$SRC_NAME
CKPT=model_2700.pt
RUN=ftsr_recovery_pd16_noslew_resume2700
test -f "$SRC/$CKPT"
it=$(uv run python -c "import torch; print(torch.load('$SRC/$CKPT', map_location='cpu', weights_only=False)['iter'])")
remaining=$((8000 - it))
echo "[resume] $(date) commit $(git rev-parse --short HEAD) from $SRC/$CKPT (iter $it), $remaining iterations"

# The new run dir inherits the best no-assist checkpoint record (model_2500).
(
  for _ in $(seq 600); do
    d=$(ls -d logs/rsl_rl/minipi_ftsr_ref/*_"$RUN" 2>/dev/null | sort | tail -1 || true)
    if [ -n "$d" ]; then
      mkdir -p "$d/eval"
      cp -n "$SRC"/best_recovery.json "$SRC"/best_recovery_model.pt "$d"/
      cp -n "$SRC"/eval/eval_*.json "$d"/eval/
      echo "[resume] best record and earlier evaluations copied to $d"
      break
    fi
    sleep 1
  done
) &

uv run train Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless --env.scene.num-envs 4000 \
  --agent.resume True --agent.load-run "$SRC_NAME" --agent.load-checkpoint "$CKPT" \
  --agent.max-iterations "$remaining" --agent.run-name "$RUN" 2>&1 \
  | tee "logs/$RUN.log"
wait
echo "[resume] $(date) done"
