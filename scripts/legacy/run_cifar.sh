#!/usr/bin/env bash
# CIFAR-10 reproduction at heavily-reduced config (9-layer probabilistic CNN is ~112s/epoch,
# CPU-bound on TITAN Xp). CIFAR's role = the "certificates degrade toward vacuity at high
# difficulty" data point, so reduced precision is acceptable + documented. Resume-safe.
set -u
cd /root/autodl-tmp/pacbayes-cert
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
export PYTHONPATH=third_party/PBB:${PYTHONPATH:-}
RESULTS=results/results_summary.csv
TRAIN_EP=${TRAIN_EP:-25}; PRIOR_EP=${PRIOR_EP:-20}; MC=${MC:-1000}; LR=${LR:-0.005}; SEED=${SEED:-0}
log(){ echo "[$(date '+%F %T')] $*"; }
run_cell(){
  local obj=$1
  local id="cifar10_${1}_learnt_cnn_seed0"
  local outdir="experiments/logs/$id"
  if [ -f "$outdir/metrics.json" ]; then log "SKIP $id (resume)"; return 0; fi
  log "START $id (prior=$PRIOR_EP post=$TRAIN_EP mc=$MC lr=$LR layers=9)"
  python scripts/train.py --dataset cifar10 --objective "$obj" --prior learnt --model cnn --layers 9 \
    --lr "$LR" --mc-samples "$MC" --train-epochs "$TRAIN_EP" --prior-epochs "$PRIOR_EP" --perc-prior 0.5 \
    --seed "$SEED" --outdir "$outdir" > "$outdir.run.log" 2>&1
  local rc=$? ok riskce risk01 stch pm ens dur
  ok=$(grep -o 'OK {.*}' "$outdir.run.log" | tail -1)
  riskce=$(echo "$ok" | grep -o '"Risk_CE": "[^"]*"' | cut -d'"' -f4)
  risk01=$(echo "$ok" | grep -o '"Risk_01": "[^"]*"' | cut -d'"' -f4)
  stch=$(echo "$ok" | grep -o '"Stch_01": "[^"]*"' | cut -d'"' -f4)
  pm=$(echo "$ok" | grep -o '"PostMean_01": "[^"]*"' | cut -d'"' -f4)
  ens=$(echo "$ok" | grep -o '"Ens_01": "[^"]*"' | cut -d'"' -f4)
  dur=$(echo "$ok" | grep -o '"duration_sec": [0-9.]*' | grep -o '[0-9.]*$')
  log "DONE $id rc=$rc Risk_CE=$riskce Risk_01=$risk01 Stch=$stch PostMean=$pm Ens=$ens (${dur}s)"
  echo "$id,$obj,learnt,cnn,$SEED,$TRAIN_EP,$MC,$rc,$riskce,$risk01,$stch,$pm,$ens,$dur" >> "$RESULTS"
}
run_cell fquad
run_cell flamb
run_cell fclassic
run_cell bbb
log "CIFAR GRID COMPLETE — see $RESULTS"
