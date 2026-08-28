#!/usr/bin/env bash
# Quick σ0 (prior-scale) screen for fquad/rand/fcn on MNIST to pick the best prior scale
# before committing the full 5h grid. 3 values run CONCURRENTLY (FCN is tiny, ~1GB each).
# Uses 70 epochs (paper: converge ~70) + mc=2000 (fast; enough to rank σ0 by error/cert).
set -u
cd /root/autodl-tmp/pacbayes-cert
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
export PYTHONPATH=third_party/PBB:${PYTHONPATH:-}
for s in 0.05 0.02 0.01; do
  python scripts/train.py --dataset mnist --objective fquad --prior rand --model fcn \
    --sigma-prior "$s" --lr 0.005 --mc-samples 2000 --train-epochs 70 --seed 0 \
    --outdir "experiments/logs/probe_sigma${s}" > "experiments/logs/probe_sigma${s}.log" 2>&1 &
done
wait
{
  echo "PROBE_DONE $(date '+%F %T')"
  for s in 0.05 0.02 0.01; do
    echo "sigma=$s $(grep -o 'OK {.*}' experiments/logs/probe_sigma${s}.log | tail -1)"
  done
} > experiments/logs/probe_summary.txt
cat experiments/logs/probe_summary.txt
