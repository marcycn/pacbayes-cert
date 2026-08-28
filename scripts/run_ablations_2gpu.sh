#!/usr/bin/env bash
# Small-data and controlled-difficulty sweeps, at the same five seeds as the
# headline matrix (the earlier study ran these at three, which the pre-registered
# protocol asks not to do for anything a published claim rests on).
#
# Both vary only the MNIST FCN f_quad learnt-prior cell; perc_train=1.0 with
# label_noise=0 is that headline cell itself and is not repeated here.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-E:/Programming/research_ws/pacbayes_env/Scripts/python.exe}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CFG=configs/mnist_fquad_learnt_fcn.yaml
RAW="$ROOT/results/raw"
mkdir -p "$RAW"

work=()
for pt in 0.5 0.2 0.1; do for s in $SEEDS; do work+=("pt:$pt:$s"); done; done
for ln in 0.1 0.2 0.4; do for s in $SEEDS; do work+=("ln:$ln:$s"); done; done

lane () {
  local gpu=$1 i=0
  for item in "${work[@]}"; do
    i=$((i+1)); [ $(( (i-1) % 2 )) -eq "$gpu" ] || continue
    IFS=':' read -r kind val seed <<< "$item"
    if [ "$kind" = "pt" ]; then
      tag="pt$(echo "$val" | tr -d '.')"; extra="--perc-train $val"
    else
      tag="ln$(echo "$val" | tr -d '.')"; extra="--label-noise $val"
    fi
    outdir="$RAW/mnist_fquad_learnt_fcn_${tag}_seed${seed}"
    [ -f "$outdir/metrics.json" ] && { echo "[gpu$gpu] SKIP ${tag}_seed${seed}"; continue; }
    mkdir -p "$outdir"
    echo "[gpu$gpu $(date '+%H:%M:%S')] START ${tag}_seed${seed}"
    t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$ROOT/src" "$PY" "$ROOT/run.py" \
         --config "$CFG" --base-seed "$seed" $extra \
         --data-root data --outdir "$outdir" >"$outdir/run.log" 2>&1; then
      echo "[gpu$gpu $(date '+%H:%M:%S')] DONE  ${tag}_seed${seed} $(( $(date +%s) - t0 ))s"
    else
      echo "[gpu$gpu $(date '+%H:%M:%S')] FAIL  ${tag}_seed${seed}"
    fi
  done
  echo "[gpu$gpu] LANE COMPLETE"
}

echo "ablations: ${#work[@]} cells over 2 GPUs, started $(date '+%F %T')"
lane 0 & p0=$!
lane 1 & p1=$!
wait $p0 $p1
echo "ABLATIONS COMPLETE $(date '+%F %T')"
