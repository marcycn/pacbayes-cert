#!/usr/bin/env bash
# Launch the experiment matrix inside a detached screen on the remote GPU so it
# survives SSH disconnects. run_matrix.sh handles resume/skip per cell.
#
# Usage (on remote):
#   SEEDS="0 1 2 3 4" CONFIGS="mnist_fquad_learnt_fcn ..." \
#     bash scripts/remote_launch.sh <session_name>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-pacbayes}"
export PATH=/root/miniconda3/bin:$PATH
PROG="$ROOT/results/${SESSION}.progress"

screen -dmS "$SESSION" bash -c "
  export PATH=/root/miniconda3/bin:\$PATH
  export SEEDS='${SEEDS:-0 1 2 3 4}'
  export CONFIGS='${CONFIGS:-}'
  export DATA_ROOT='${DATA_ROOT:-data_all}'
  export MAXJOBS='${MAXJOBS:-1}'
  bash '$ROOT/scripts/run_matrix.sh' > '$PROG' 2>&1
  echo ALL_DONE >> '$PROG'
"
echo "launched screen '$SESSION'; tail -f $PROG"
