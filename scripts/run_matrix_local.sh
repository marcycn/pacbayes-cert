#!/usr/bin/env bash
# Run the full experiment matrix on the local GPU, resumable.
#
# Sequential by design: every cell now evaluates the Monte-Carlo certificate in
# one or two very large batches, so a single run already saturates the card and
# concurrent runs would only trade throughput for OOM risk.
#
#   bash scripts/run_matrix_local.sh                    # headline matrix, 5 seeds
#   CONFIGS="mnist_fquad_learnt_fcn" SEEDS="0 1" bash scripts/run_matrix_local.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-D:/Anaconda/envs/multilingual_lora/python.exe}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIGS="${CONFIGS:-$(ls configs/*.yaml | xargs -n1 basename | sed 's/\.yaml$//')}"
DATA_ROOT="${DATA_ROOT:-data}"
RAW="$ROOT/results/raw"
mkdir -p "$RAW"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/results/matrix_local.log"; }

total=0; done_=0; failed=0
for cfg in $CONFIGS; do for seed in $SEEDS; do total=$((total+1)); done; done
log "MATRIX START: $total cells ($(echo $CONFIGS | wc -w) configs x $(echo $SEEDS | wc -w) seeds)"

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    outdir="$RAW/${cfg}_seed${seed}"
    if [ -f "$outdir/metrics.json" ]; then
      log "SKIP  ${cfg}_seed${seed} (already has metrics.json)"
      done_=$((done_+1)); continue
    fi
    mkdir -p "$outdir"
    log "START ${cfg}_seed${seed}"
    t0=$(date +%s)
    if "$PY" "$ROOT/run.py" --config "configs/${cfg}.yaml" --base-seed "$seed" \
         --data-root "$DATA_ROOT" --outdir "$outdir" >"$outdir/run.log" 2>&1; then
      t=$(( $(date +%s) - t0 ))
      cert=$("$PY" -c "import json;print(round(json.load(open(r'$outdir/metrics.json'))['cert_risk_01'],4))" 2>/dev/null || echo "?")
      log "DONE  ${cfg}_seed${seed}  cert=$cert  ${t}s"
      done_=$((done_+1))
    else
      log "FAIL  ${cfg}_seed${seed} (see $outdir/run.log)"
      failed=$((failed+1))
    fi
  done
done
log "MATRIX COMPLETE: $done_ ok, $failed failed, of $total"
