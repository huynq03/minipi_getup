#!/usr/bin/env bash
# Zero-shot limiter evaluation (evaluation only): 4 checkpoints x {baseline, 0.3} x seeds.
# Two rollouts at a time. Skips runs that already have stats.json.
set -u
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
export WARP_CACHE_PATH=/home/huy/minipi_getup/.warp-cache-cu12
RUN=logs/rsl_rl/minipi_ftsr_ref/2026-10-09_08-33-09_ftsr_recovery_pd16_noslew_resume2700
OUT=results/ftsr_limiter_0p3_evaluation
SEEDS=${SEEDS:-"2150 2151 2152"}
mkdir -p $OUT/runs $OUT/logs
jobs_list=()
for s in $SEEDS; do
  for it in 3000 4000 5000 8000; do
    for L in 0 0.3; do
      c=baseline; [ "$L" != 0 ] && c=limited
      jobs_list+=("$it $c $L $s")
    done
  done
done
run_one() {
  set -- $1
  local d=$OUT/runs/it$1_$2_s$4
  [ -f $d/stats.json ] && { echo "skip $d"; return; }
  .venv/bin/python -m minipi_getup.ftsr_ref.limiter_eval rollout \
    --checkpoint $RUN/model_$1.pt --limit $3 --seed $4 --per-pose 128 --out $d \
    > $OUT/logs/it$1_$2_s$4.log 2>&1
  echo "$(date +%T) done $d rc=$? $(tail -1 $OUT/logs/it$1_$2_s$4.log)"
}
i=0
for j in "${jobs_list[@]}"; do
  run_one "$j" &
  i=$((i + 1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
echo "ALL ROLLOUTS DONE"
