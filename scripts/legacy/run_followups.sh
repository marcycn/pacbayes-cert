#!/usr/bin/env bash
# Dissertation-mark follow-ups on the GPU (resume-safe, appends to results_summary.csv):
#  (A) MULTI-SEED: MNIST FCN learnt {fquad,flamb,fclassic} seeds 1,2 (seed 0 exists) -> CIs.
#  (B) SMALL-DATA: MNIST fquad learnt fcn at perc_train {0.5,0.2,0.1} (1.0 exists) -> data-efficiency curve.
set -u
cd /root/autodl-tmp/pacbayes-cert
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
export PYTHONPATH=third_party/PBB:${PYTHONPATH:-}
RESULTS=results/results_summary.csv
MC=10000; EP=100; LR=0.005
log(){ echo "[$(date '+%F %T')] $*"; }
run(){
  local obj=$1 prior=$2 model=$3 seed=$4 pt=$5 id=$6 outdir="experiments/logs/$6"
  if [ -f "$outdir/metrics.json" ]; then log "SKIP $id"; return 0; fi
  log "START $id (seed=$seed perc_train=$pt)"
  python scripts/train.py --dataset mnist --objective "$obj" --prior "$prior" --model "$model" \
    --mc-samples $MC --train-epochs $EP --seed "$seed" --lr $LR --perc-train "$pt" \
    --outdir "$outdir" > "$outdir.run.log" 2>&1
  local rc=$? ok risk01 stch pm ens dur
  ok=$(grep -o 'OK {.*}' "$outdir.run.log" | tail -1)
  risk01=$(echo "$ok"|grep -o '"Risk_01": "[^"]*"'|cut -d'"' -f4)
  stch=$(echo "$ok"|grep -o '"Stch_01": "[^"]*"'|cut -d'"' -f4)
  pm=$(echo "$ok"|grep -o '"PostMean_01": "[^"]*"'|cut -d'"' -f4)
  ens=$(echo "$ok"|grep -o '"Ens_01": "[^"]*"'|cut -d'"' -f4)
  dur=$(echo "$ok"|grep -o '"duration_sec": [0-9.]*'|grep -o '[0-9.]*$')
  log "DONE $id rc=$rc Risk_01=$risk01 Stch=$stch PostMean=$pm Ens=$ens (${dur}s)"
  echo "$id,$obj,$prior,$model,$seed,$EP,$MC,$rc,,$risk01,$stch,$pm,$ens,$dur" >> "$RESULTS"
}
# (A) multi-seed on the objective comparison (learnt prior, MNIST FCN)
for s in 1 2; do
  for o in fquad flamb fclassic; do
    run "$o" learnt fcn "$s" 1.0 "mnist_${o}_learnt_fcn_seed${s}"
  done
done
# (B) small-data self-certified sweep (MNIST fquad learnt fcn, seed 0)
for pt in 0.5 0.2 0.1; do
  run fquad learnt fcn 0 "$pt" "mnist_fquad_learnt_fcn_seed0_pt${pt}"
done
log "FOLLOWUPS COMPLETE — see $RESULTS"
