#!/usr/bin/env bash
# Run the experiment matrix across seeds, resumable, with bounded parallelism.
# FCN cells are CPU/overhead-bound and use little GPU, so MAXJOBS run concurrently.
# Portable root resolution (P2 §5.2), strict-ish mode (P2 §5.3), per-seed dirs (P2 §5.4).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/root/miniconda3/bin:${PATH:-}
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIGS="${CONFIGS:-$(ls configs/*.yaml | xargs -n1 basename | sed 's/.yaml//')}"
DATA_ROOT="${DATA_ROOT:-data_all}"
MAXJOBS="${MAXJOBS:-1}"
RAW="$ROOT/results/raw"
mkdir -p "$RAW"
log(){ echo "[$(date '+%F %T')] $*"; }

run_one(){
  local cfg=$1 seed=$2
  local outdir="$RAW/${cfg}_seed${seed}"
  [ -f "$outdir/metrics.json" ] && { log "SKIP ${cfg}_seed${seed}"; return 0; }
  log "START ${cfg}_seed${seed}"
  if python "$ROOT/run.py" --config "configs/${cfg}.yaml" --base-seed "$seed" \
       --data-root "$DATA_ROOT" --outdir "$outdir" >"$outdir.log" 2>&1; then
    log "DONE  ${cfg}_seed${seed}"
  else
    log "FAIL  ${cfg}_seed${seed} (rc=$? see $outdir.log)"
  fi
}

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
    run_one "$cfg" "$seed" &
  done
done
wait
log "MATRIX COMPLETE"
