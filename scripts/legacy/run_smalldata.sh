#!/usr/bin/env bash
# Small-data self-certified ablation (valid under the codebase's fixed seed — varies perc_train,
# not the seed): MNIST fquad learnt fcn at perc_train {0.5,0.2,0.1} (1.0 already in CSV).
# Resume-safe; appends to results_summary.csv with perc_train encoded in the run id.
set -u
cd /root/autodl-tmp/pacbayes-cert
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
export PYTHONPATH=third_party/PBB:${PYTHONPATH:-}
RESULTS=results/results_summary.csv
MC=10000; EP=100; LR=0.005; SEED=0
log(){ echo "[$(date '+%F %T')] $*"; }
run(){
  local pt=$1
  local id="mnist_fquad_learnt_fcn_seed0_pt${1}"
  local outdir="experiments/logs/$id"
  if [ -f "$outdir/metrics.json" ]; then log "SKIP $id"; return 0; fi
  log "START $id (perc_train=$pt)"
  python scripts/train.py --dataset mnist --objective fquad --prior learnt --model fcn \
    --mc-samples $MC --train-epochs $EP --seed $SEED --lr $LR --perc-train "$pt" \
    --outdir "$outdir" > "$outdir.run.log" 2>&1
  local rc=$? ok risk01 stch pm ens dur
  ok=$(grep -o 'OK {.*}' "$outdir.run.log" | tail -1)
  risk01=$(echo "$ok"|grep -o '"Risk_01": "[^"]*"'|cut -d'"' -f4)
  stch=$(echo "$ok"|grep -o '"Stch_01": "[^"]*"'|cut -d'"' -f4)
  pm=$(echo "$ok"|grep -o '"PostMean_01": "[^"]*"'|cut -d'"' -f4)
  ens=$(echo "$ok"|grep -o '"Ens_01": "[^"]*"'|cut -d'"' -f4)
  dur=$(echo "$ok"|grep -o '"duration_sec": [0-9.]*'|grep -o '[0-9.]*$')
  log "DONE $id rc=$rc Risk_01=$risk01 Stch=$stch PostMean=$pm Ens=$ens (${dur}s)"
  echo "$id,fquad,learnt,fcn,$SEED,$EP,$MC,$rc,,$risk01,$stch,$pm,$ens,$dur" >> "$RESULTS"
}
for pt in 0.5 0.2 0.1; do run "$pt"; done
log "SMALLDATA COMPLETE — see $RESULTS"
