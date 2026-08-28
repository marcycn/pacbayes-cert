#!/usr/bin/env bash
# Parallel matrix runner: the FCN cells are CPU/overhead-bound and use little GPU,
# so many run concurrently on one GPU. Concurrency is bounded by MAXJOBS; per-process
# CPU threads are capped to avoid oversubscription on the 56-core host.
#
# Usage (on remote, inside screen):
#   MAXJOBS=12 SEEDS="0 1 2 3 4" CONFIGS="<list>" DATA_ROOT=data_all \
#     bash scripts/run_parallel.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

MAXJOBS="${MAXJOBS:-12}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIGS="${CONFIGS:?set CONFIGS}"
DATA_ROOT="${DATA_ROOT:-data_all}"
mkdir -p "$ROOT/results/raw"
log(){ echo "[$(date '+%F %T')] $*"; }

run_one(){
  local cfg=$1 seed=$2
  local outdir="$ROOT/results/raw/${cfg}_seed${seed}"
  [ -f "$outdir/metrics.json" ] && { log "SKIP ${cfg}_seed${seed}"; return 0; }
  log "START ${cfg}_seed${seed}"
  if python "$ROOT/run.py" --config "$ROOT/configs/${cfg}.yaml" --base-seed "$seed" \
       --data-root "$DATA_ROOT" --outdir "$outdir" >"$outdir.log" 2>&1; then
    log "DONE  ${cfg}_seed${seed}"
  else
    log "FAIL  ${cfg}_seed${seed} (rc=$? see $outdir.log)"
  fi
}

jobs_running(){ jobs -rp | wc -l; }

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    while [ "$(jobs_running)" -ge "$MAXJOBS" ]; do sleep 3; done
    run_one "$cfg" "$seed" &
  done
done
wait
log "PARALLEL MATRIX COMPLETE"
