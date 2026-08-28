#!/usr/bin/env bash
# Reproduce JMLR 2021 (arXiv:2007.12911) Table 1 on MNIST.
# Cells run sequentially; each -> experiments/logs/<id>/ + a row in results/results_summary.csv.
# Tune via env: TRAIN_EPOCHS (default 100), MC (mc_samples, default 10000), SEED (default 0).
set -u
cd /root/autodl-tmp/pacbayes-cert
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
export PYTHONPATH=third_party/PBB:${PYTHONPATH:-}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}
MC=${MC:-10000}
SEED=${SEED:-0}
LR=${LR:-0.005}        # paper §7.2.1: posterior LR 1e-3 converged slowly; 5e-3 is the sweet spot (1e-2 diverges)
RESULTS=results/results_summary.csv
mkdir -p results experiments/logs
[ -f "$RESULTS" ] || echo "run_id,objective,prior,model,seed,train_epochs,mc_samples,rc,risk_ce,risk_01,stch_01,postmean_01,ens_01,duration_sec" >> "$RESULTS"
log(){ echo "[$(date '+%F %T')] $*"; }

run_cell(){
  local obj=$1 prior=$2 model=$3
  local id="mnist_${obj}_${prior}_${model}_seed${SEED}"
  local outdir="experiments/logs/$id"
  if [ -f "$outdir/metrics.json" ]; then log "SKIP $id (metrics.json exists — resume)"; return 0; fi
  log "START $id (epochs=$TRAIN_EPOCHS mc=$MC)"
  python scripts/train.py --dataset mnist --objective "$obj" --prior "$prior" --model "$model" \
    --mc-samples "$MC" --train-epochs "$TRAIN_EPOCHS" --seed "$SEED" --outdir "$outdir" --lr "$LR" \
    > "$outdir.run.log" 2>&1
  local rc=$? ok riskce risk01 stch pm ens dur
  ok=$(grep -o 'OK {.*}' "$outdir.run.log" | tail -1)
  riskce=$(echo "$ok" | grep -o '"Risk_CE": "[^"]*"' | cut -d'"' -f4)
  risk01=$(echo "$ok" | grep -o '"Risk_01": "[^"]*"' | cut -d'"' -f4)
  stch=$(echo "$ok" | grep -o '"Stch_01": "[^"]*"' | cut -d'"' -f4)
  pm=$(echo "$ok" | grep -o '"PostMean_01": "[^"]*"' | cut -d'"' -f4)
  ens=$(echo "$ok" | grep -o '"Ens_01": "[^"]*"' | cut -d'"' -f4)
  dur=$(echo "$ok" | grep -o '"duration_sec": [0-9.]*' | grep -o '[0-9.]*$')
  log "DONE $id rc=$rc  Risk_CE=$riskce Risk_01=$risk01 Stch=$stch PostMean=$pm Ens=$ens (${dur}s)"
  echo "$id,$obj,$prior,$model,$SEED,$TRAIN_EPOCHS,$MC,$rc,$riskce,$risk01,$stch,$pm,$ens,$dur" >> "$RESULTS"
}

# FCN + data-free (rand) prior
run_cell fquad    rand fcn
run_cell flamb    rand fcn
run_cell fclassic rand fcn
run_cell bbb      rand fcn
# FCN + data-dependent (learnt) prior  (prior_epochs=70, perc_prior=0.5 via train.py defaults)
run_cell fquad    learnt fcn
run_cell flamb    learnt fcn
run_cell fclassic learnt fcn
run_cell bbb      learnt fcn
# CNN + learnt prior
run_cell fquad    learnt cnn
run_cell flamb    learnt cnn
run_cell fclassic learnt cnn
log "GRID COMPLETE — see $RESULTS"
