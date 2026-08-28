#!/usr/bin/env bash
# Generalized PBB grid runner — works for any dataset whose branch exists in pbb/data.py
# (mnist, fashion-mnist, cifar10). Usage: DATASET=fashion-mnist [TRAIN_EPOCHS=100 MC=10000 LR=0.005 SEED=0] bash run_grid_any.sh
# Resume-safe: skips cells whose metrics.json already exists.
set -u
cd /root/autodl-tmp/pacbayes-cert
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
export PYTHONPATH=third_party/PBB:${PYTHONPATH:-}
DATASET=${DATASET:-mnist}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}
MC=${MC:-10000}
# CNN cells use a reduced config — probabilistic-CNN is ~10× slower/step (CPU-bound on TITAN Xp)
EPOCHS_CNN=${EPOCHS_CNN:-50}
MC_CNN=${MC_CNN:-2000}
SEED=${SEED:-0}
LR=${LR:-0.005}
# FCN cells run for mnist/fashion-mnist; CIFAR-10 uses CNN (9-layer) — pick models per dataset
RESULTS=results/results_summary.csv
mkdir -p results experiments/logs
[ -f "$RESULTS" ] || echo "run_id,objective,prior,model,seed,train_epochs,mc_samples,rc,risk_ce,risk_01,stch_01,postmean_01,ens_01,duration_sec" >> "$RESULTS"
log(){ echo "[$(date '+%F %T')] $*"; }

run_cell(){
  local obj=$1 prior=$2 model=$3 layers=${4:-9}
  local id="${DATASET}_${obj}_${prior}_${model}_seed${SEED}"
  local outdir="experiments/logs/$id"
  if [ -f "$outdir/metrics.json" ]; then log "SKIP $id (resume)"; return 0; fi
  local eff_mc=$MC eff_ep=$TRAIN_EPOCHS
  [ "$model" = "cnn" ] && { eff_mc=$MC_CNN; eff_ep=$EPOCHS_CNN; }
  log "START $id (epochs=$eff_ep mc=$eff_mc lr=$LR layers=$layers)"
  local extra=""; [ "$DATASET" = "cifar10" ] && extra="--layers $layers"
  python scripts/train.py --dataset "$DATASET" --objective "$obj" --prior "$prior" --model "$model" \
    --mc-samples "$eff_mc" --train-epochs "$eff_ep" --seed "$SEED" --outdir "$outdir" --lr "$LR" $extra \
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
  echo "$id,$obj,$prior,$model,$SEED,$eff_ep,$eff_mc,$rc,$riskce,$risk01,$stch,$pm,$ens,$dur" >> "$RESULTS"
}

case "$DATASET" in
  mnist|fashion-mnist)
    run_cell fquad    rand fcn
    run_cell flamb    rand fcn
    run_cell fclassic rand fcn
    run_cell bbb      rand fcn
    run_cell fquad    learnt fcn
    run_cell flamb    learnt fcn
    run_cell fclassic learnt fcn
    run_cell bbb      learnt fcn
    run_cell fquad    learnt cnn
    run_cell flamb    learnt cnn
    run_cell fclassic learnt cnn
    ;;
  cifar10)
    run_cell fquad    learnt cnn 9
    run_cell flamb    learnt cnn 9
    run_cell fclassic learnt cnn 9
    run_cell bbb      learnt cnn 9
    ;;
esac
log "GRID COMPLETE ($DATASET) — see $RESULTS"
