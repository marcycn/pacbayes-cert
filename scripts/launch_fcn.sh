#!/usr/bin/env bash
# Self-contained launcher for the FCN matrix (MNIST + Fashion-MNIST), parallel.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/root/miniconda3/bin:$PATH
export CONFIGS="mnist_fquad_learnt_fcn mnist_fclassic_learnt_fcn mnist_bbb_learnt_fcn mnist_flamb_learnt_fcn mnist_fquad_rand_fcn fashion-mnist_fquad_learnt_fcn fashion-mnist_fclassic_learnt_fcn fashion-mnist_bbb_learnt_fcn fashion-mnist_fquad_rand_fcn"
export SEEDS="${SEEDS:-0 1 2 3 4}"
export MAXJOBS="${MAXJOBS:-14}"
export DATA_ROOT="${DATA_ROOT:-data_all}"
bash "$ROOT/scripts/run_parallel.sh" > "$ROOT/results/fcn.progress" 2>&1
echo ALLDONE >> "$ROOT/results/fcn.progress"
