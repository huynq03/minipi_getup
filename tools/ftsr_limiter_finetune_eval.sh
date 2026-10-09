#!/usr/bin/env bash
# Evaluation of the limiter fine-tune (evaluation only): source model_4000 and the
# fine-tuned checkpoints, all on the limiter task (limiter from the task cfg, applied
# once), deterministic student, zero assistance, seeds 2150-2152 x 128 per pose.
# Two rollouts at a time; skips runs that already have stats.json.
set -u
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
export WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12
TASK=Mjlab-FTSR-Ref-MiniPi-Recovery-Stateless-Limit0p3
SRC=logs/rsl_rl/minipi_ftsr_ref/2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700
FT=${FT:?set FT=run dir of the fine-tune}
OUT=${OUT:-results/ftsr_limiter_0p3_finetune}
ITERS=${ITERS:-"4250 4500 4750 5000"}
SEEDS=${SEEDS:-"2150 2151 2152"}
mkdir -p $OUT/runs $OUT/logs
jobs_list=()
for s in $SEEDS; do
  jobs_list+=("$SRC/model_4000.pt 4000 $s")
  for it in $ITERS; do jobs_list+=("$FT/model_$it.pt $it $s"); done
done
run_one() {
  set -- $1
  local d=$OUT/runs/it$2_limited_s$3
  [ -f $d/stats.json ] && { echo "skip $d"; return; }
  .venv/bin/python -m minipi_getup.ftsr_ref.limiter_eval rollout --task $TASK \
    --checkpoint $1 --limit 0 --seed $3 --per-pose 128 --out $d \
    > $OUT/logs/it$2_s$3.log 2>&1
  echo "$(date +%T) done $d rc=$? $(tail -1 $OUT/logs/it$2_s$3.log)"
}
i=0
for j in "${jobs_list[@]}"; do
  run_one "$j" &
  i=$((i + 1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
echo "ALL ROLLOUTS DONE"
