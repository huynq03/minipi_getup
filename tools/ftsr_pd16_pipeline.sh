#!/usr/bin/env bash
# Experiment pd16_noslew: walking initialization from scratch, its evaluation and
# acceptance check, then FTSR recovery (v2 stateless stages) from its model_900.
# Run inside tmux from the repository root. Never touches hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
export WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12
mkdir -p logs

WALK_RUN=ftsr_walk_pd16_noslew
REC_RUN=ftsr_recovery_pd16_noslew
ENVS=4000

echo "[pipeline] $(date) commit $(git rev-parse --short HEAD) walk training"
uv run train Mjlab-FTSR-Ref-MiniPi-Walk --env.scene.num-envs "$ENVS" \
  --agent.max-iterations 900 --agent.run-name "$WALK_RUN" 2>&1 | tee "logs/$WALK_RUN.log"

WALK_DIR=$(ls -d logs/rsl_rl/minipi_ftsr_ref_walk/*_"$WALK_RUN" | sort | tail -1)
CKPT="$WALK_DIR/model_900.pt"
test -f "$CKPT"
echo "[pipeline] $(date) walk evaluation of $CKPT"
uv run python -m minipi_getup.ftsr_ref.evaluate --checkpoint "$CKPT" \
  --task Mjlab-FTSR-Ref-MiniPi-Walk --out "$WALK_DIR/eval_walk_900.json" \
  > "$WALK_DIR/eval_walk_900.log" 2>&1

# Acceptance (walking initialization, paper Sec. III-A "elementary walking"):
# vx tracking correlation >= 0.9 and fall rate <= 10 %; physical metrics reported only.
uv run python - "$WALK_DIR/eval_walk_900.json" <<'EOF'
import json, math, sys
d = json.load(open(sys.argv[1]))
p = d["physical"]
ok = (d["corr_vx"] is not None and d["corr_vx"] >= 0.9 and d["fall_rate"] <= 0.10
      and all(math.isfinite(v) for v in (d["corr_vx"], d["fall_rate"])))
print(f"[pipeline] walk: corr_vx {d['corr_vx']:.3f} corr_wz {d['corr_wz']} "
      f"fall_rate {d['fall_rate']:.3f} | qd p99 max {max(p['qd_p99'].values()):.2f} "
      f">6.28 {p['qd_frac_above_6p28']:.4f} | tau max {max(p['tau_max'].values()):.1f} "
      f"near cap {p['tau_frac_near_cap']:.4f} | limit margin {p['joint_limit_margin_min']:.3f}"
      f" -> {'ACCEPTED' if ok else 'REJECTED'}")
json.dump({"accepted": ok}, open(sys.argv[1].replace(".json", "_accept.json"), "w"))
sys.exit(0 if ok else 3)
EOF

echo "[pipeline] $(date) recovery training from $CKPT"
uv run train Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless --env.scene.num-envs "$ENVS" \
  --agent.init-checkpoint "$(realpath "$CKPT")" --agent.run-name "$REC_RUN" 2>&1 \
  | tee "logs/$REC_RUN.log"
echo "[pipeline] $(date) done"
