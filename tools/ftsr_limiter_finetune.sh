#!/usr/bin/env bash
# Limiter fine-tune: resume FTSR recovery from model_4000.pt of the pd16_noslew run
# (resume2700) on Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3, the stateless
# task with a 0.3 rad / policy-step PD-target rate limiter. Restores networks (actor,
# critic, teacher and student encoders, std), both Adam states, adaptive lr, iteration,
# env step counter (assist schedule: t > t_tag, assistance stays zero) and RNG; trains
# 1000 ADDITIONAL iterations (4000 -> 5000) in a new run dir. Nothing else changes.
# Run inside tmux from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
export WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12

TASK=Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3
SRC_NAME=2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700
SRC=logs/rsl_rl/minipi_ftsr_ref/$SRC_NAME
CKPT=model_4000.pt
SHA=408df87e06b9fa49659fd98fb971c18fcc27814c53d5d164e97c4b4e43d20ad3
RUN=ftsr_limit0p3_ft4000
ADD=1000
test -f "$SRC/$CKPT"
echo "$SHA  $SRC/$CKPT" | sha256sum -c -
if [ -n "$(git status --porcelain -- src tools)" ]; then
  echo "[finetune] uncommitted changes in src/ or tools/; commit first" >&2
  exit 1
fi
it=$(.venv/bin/python -c "import torch; print(torch.load('$SRC/$CKPT', map_location='cpu', weights_only=False)['iter'])")
echo "[finetune] $(date) commit $(git rev-parse --short HEAD) task $TASK from $SRC/$CKPT (iter $it), +$ADD iterations"

# Provenance record in the new run dir (once it exists).
(
  for _ in $(seq 900); do
    d=$(ls -d logs/rsl_rl/minipi_ftsr_ref/*_"$RUN" 2>/dev/null | sort | tail -1 || true)
    if [ -n "$d" ]; then
      .venv/bin/python - "$d" <<EOF
import json, subprocess, sys
d = sys.argv[1]
rec = {
  "experiment": "limiter fine-tune 0.3 rad/policy step (adaptation of an FTSR policy)",
  "task": "$TASK",
  "source_run": "$SRC",
  "source_checkpoint": "$SRC/$CKPT",
  "source_sha256": "$SHA",
  "source_iteration": $it,
  "additional_iterations": $ADD,
  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
  "changes_vs_source_task": {
    "env.actions.joint_pos.target_rate_limit": [0.0, 0.3],
    "agent.eval_every": [500, 250],
    "agent.eval_at": ["(500, ..., 3500)", "()"],
    "agent.eval_task": ["Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless", "$TASK"],
  },
  "unchanged": "rewards, entropy, lr schedule, std, PPO, teacher/student split, "
  "force guidance (assist schedule ended at it 3000), observations, commands, reset "
  "poses, passive window, PD gains, 16 Nm cap, joint ranges, physics",
  "best_record": "not inherited: best_recovery.json here ranks limited evaluations only",
}
json.dump(rec, open(f"{d}/finetune_source.json", "w"), indent=1)
EOF
      echo "[finetune] provenance written to $d/finetune_source.json"
      break
    fi
    sleep 1
  done
) &

uv run train "$TASK" --env.scene.num-envs 4000 \
  --agent.resume True --agent.load-run "$SRC_NAME" --agent.load-checkpoint "$CKPT" \
  --agent.max-iterations "$ADD" --agent.run-name "$RUN" 2>&1 \
  | tee "logs/$RUN.log"
wait
echo "[finetune] $(date) done"
