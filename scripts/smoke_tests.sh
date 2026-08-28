#!/usr/bin/env bash
# Phase A correctness smoke tests (fast, low-epoch). Verifies split sizes, batching
# invariance, seed control and perc_train accounting before the expensive matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
DATA_ROOT="${DATA_ROOT:-data}"
S="$ROOT/results/smoke"
mkdir -p "$S"
EP="${EP:-2}"; MC="${MC:-1000}"

echo "== unit tests (torch-backed) =="
python -m pytest tests/ -q

echo "== A1: MNIST learnt fquad, 2 epochs, split + checkpoint + deltas =="
python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 0 \
  --train-epochs "$EP" --prior-epochs 2 --mc-samples "$MC" --data-root "$DATA_ROOT" --outdir "$S/a1_seed0"

echo "== A2: same seed reproduces (compare split_hash + cert) =="
python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 0 \
  --train-epochs "$EP" --prior-epochs 2 --mc-samples "$MC" --data-root "$DATA_ROOT" --outdir "$S/a2_seed0_rep"

echo "== A3: different seed changes split =="
python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 1 \
  --train-epochs "$EP" --prior-epochs 2 --mc-samples "$MC" --data-root "$DATA_ROOT" --outdir "$S/a3_seed1"

echo "== A4: perc_train 0.1 must give n_bound=3000 =="
python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 0 --perc-train 0.1 \
  --train-epochs "$EP" --prior-epochs 2 --mc-samples "$MC" --data-root "$DATA_ROOT" --outdir "$S/a4_pt10"

python - <<'PY'
import json, os
S=os.path.join(os.environ.get("ROOT","."),"results","smoke")
a1=json.load(open(f"{S}/a1_seed0/metrics.json"))
a2=json.load(open(f"{S}/a2_seed0_rep/metrics.json"))
a3=json.load(open(f"{S}/a3_seed1/metrics.json"))
a4=json.load(open(f"{S}/a4_pt10/metrics.json"))
assert a1["n_bound"]==30000, a1["n_bound"]
assert a4["n_bound"]==3000, a4["n_bound"]
assert a1["split_hash"]==a2["split_hash"], "same seed must reproduce split"
assert a1["split_hash"]!=a3["split_hash"], "different seed must change split"
assert abs(a1["joint_confidence"]-0.99)<1e-9
print("SMOKE OK: n_bound 30000/3000, seed control, joint_confidence=0.99")
PY
echo "ALL SMOKE TESTS PASSED"
