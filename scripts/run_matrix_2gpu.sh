#!/usr/bin/env bash
# Split the experiment matrix across the two local GPUs and run it, resumable.
#
# Cells are interleaved round-robin so the two halves finish at about the same
# time even though CIFAR cells cost far more than FCN ones.
#
#   bash scripts/run_matrix_2gpu.sh                 # everything, 5 seeds
#   CONFIGS="mnist_fquad_learnt_fcn" bash scripts/run_matrix_2gpu.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-E:/Programming/research_ws/pacbayes_env/Scripts/python.exe}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIGS="${CONFIGS:-$(ls configs/*.yaml | xargs -n1 basename | sed 's/\.yaml$//')}"
DATA_ROOT="${DATA_ROOT:-data}"
RAW="$ROOT/results/raw"
mkdir -p "$RAW"

# build the full work list, then deal it alternately to the two GPUs
work=()
for cfg in $CONFIGS; do for s in $SEEDS; do work+=("$cfg:$s"); done; done

half () {  # $1 = gpu id
  local gpu=$1 i=0
  for item in "${work[@]}"; do
    i=$((i+1))
    [ $(( (i-1) % 2 )) -eq "$gpu" ] || continue
    local cfg="${item%%:*}" seed="${item##*:}"
    local outdir="$RAW/${cfg}_seed${seed}"
    if [ -f "$outdir/metrics.json" ]; then
      echo "[gpu$gpu] SKIP  ${cfg}_seed${seed}"; continue
    fi
    mkdir -p "$outdir"
    echo "[gpu$gpu $(date '+%H:%M:%S')] START ${cfg}_seed${seed}"
    local t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$ROOT/src" "$PY" "$ROOT/run.py" \
         --config "configs/${cfg}.yaml" --base-seed "$seed" \
         --data-root "$DATA_ROOT" --outdir "$outdir" >"$outdir/run.log" 2>&1; then
      local cert=$(PYTHONPATH="$ROOT/src" "$PY" -c "import json;print(round(json.load(open(r'$outdir/metrics.json'))['cert_risk_01'],4))" 2>/dev/null || echo '?')
      echo "[gpu$gpu $(date '+%H:%M:%S')] DONE  ${cfg}_seed${seed} cert=$cert $(( $(date +%s) - t0 ))s"
    else
      echo "[gpu$gpu $(date '+%H:%M:%S')] FAIL  ${cfg}_seed${seed} (see $outdir/run.log)"
    fi
  done
  echo "[gpu$gpu] LANE COMPLETE"
}

echo "matrix: ${#work[@]} cells over 2 GPUs, started $(date '+%F %T')"
half 0 & p0=$!
half 1 & p1=$!
wait $p0 $p1
echo "MATRIX COMPLETE $(date '+%F %T')"
