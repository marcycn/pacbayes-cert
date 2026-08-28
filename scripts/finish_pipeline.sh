#!/usr/bin/env bash
# Everything that has to happen after the headline matrix, in order, unattended.
#
# The label-independent split invalidated every learnt-prior run, so the
# ablations need redoing too, and the Monte-Carlo sweep and per-class Gibbs
# analysis both read a checkpoint and so must follow the runs that produce it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-E:/Programming/research_ws/pacbayes_env/Scripts/python.exe}"
export PYTHONPATH="$ROOT/src"
log(){ echo "[$(date '+%F %T')] $*"; }

log "waiting for the headline matrix"
while ! grep -q "MATRIX COMPLETE" results/matrix_rerun.log 2>/dev/null; do sleep 60; done
log "matrix done: $(grep -c DONE results/matrix_rerun.log) ok, $(grep -c FAIL results/matrix_rerun.log) failed"

log "ablations"
bash scripts/run_ablations_2gpu.sh > results/ablations_rerun.log 2>&1
log "ablations done: $(grep -c DONE results/ablations_rerun.log) ok, $(grep -c FAIL results/ablations_rerun.log) failed"

log "Monte-Carlo budget sweep, measured and analytic"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/mc_sensitivity.py \
  --run-dir results/raw/mnist_fquad_learnt_fcn_seed0 \
  --mc-list 1000,2000,5000,10000,50000,150000 --with-analytic \
  --data-root data > results/mc_sens_rerun.log 2>&1
log "mc sweep done"

log "per-class Gibbs risk on the cells the confusion figure uses"
for d in fashion-mnist_fquad_learnt_fcn_seed0 cifar10_fquad_learnt_cnn_seed0 \
         cifar10_fquad_learnt_cnn13_seed0 mnist_fquad_learnt_fcn_seed0; do
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/gibbs_per_class.py --run-dir "results/raw/$d" \
    --data-root data 2>&1 | grep -E "Gibbs overall|worst"
done

log "aggregating and regenerating tables and figures"
"$PY" scripts/aggregate.py
"$PY" scripts/gen_tables.py
"$PY" scripts/make_figures.py > results/figures_rerun.log 2>&1 || log "make_figures reported an error"
log "PIPELINE COMPLETE"
