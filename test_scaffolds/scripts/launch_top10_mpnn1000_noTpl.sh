#!/usr/bin/env bash
# Top9：与 test_scaffords_5.20_1000 相同 MSA（mpnn1000_top），**无 binder 结构模板**
# 对照 _1000（带 mask.pdb template）→ 输出到 test_scaffords_5.20_1000_noTpl
#
#   FORCE=1 SKIP_GPU_WAIT=1 bash launch_top10_mpnn1000_noTpl.sh --tmux-9
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/data/wenhao/Protein-Hunter-server"
INPUT_ROOT="${INPUT_ROOT:-$REPO/test_scaffords_5.20_1000}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/test_scaffords_5.20_1000_noTpl}"
TAG="${TAG:-_mpnn1000_top}"
MSA_N=1000
MSA_MAX=1000

export USE_BINDER_TEMPLATE=0
export ALLOW_QUERY_ALLX=0
export MSA_N MSA_MAX TAG INPUT_ROOT RESULTS_ROOT
export SESSION="${SESSION:-boltz_top9_mpnn1000_noTpl_gpu567}"

mkdir -p "$RESULTS_ROOT"
FORCE="${FORCE:-1}" SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-1}" \
  bash "$SCRIPT_DIR/launch_top10_hard_boltz_mpnn1500.sh" --tmux-9
