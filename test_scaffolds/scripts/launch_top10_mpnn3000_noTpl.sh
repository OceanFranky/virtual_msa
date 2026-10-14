#!/usr/bin/env bash
# Top9：mpnn3000_unique（3000 条 MSA）+ a3m query，**无 binder 结构模板**（MSA 仍用 precomputed）
# 对照：test_scaffords_5.20_250（250+模板）等；输出 test_scaffords_5.20_3000_noTpl
#
#   FORCE=1 SKIP_GPU_WAIT=1 bash launch_top10_mpnn3000_noTpl.sh --tmux-8gpu
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/data/wenhao/Protein-Hunter-server"
INPUT_ROOT="${INPUT_ROOT:-$REPO/test_scaffords_5.20_3000_noTpl}"
RESULTS_ROOT="${RESULTS_ROOT:-$INPUT_ROOT}"
TAG="${TAG:-_mpnn3000_unique}"
MSA_N=3000
MSA_MAX=3000

export USE_BINDER_TEMPLATE=0
export ALLOW_QUERY_ALLX=0
export MSA_N MSA_MAX TAG INPUT_ROOT RESULTS_ROOT
export SESSION="${SESSION:-boltz_top9_mpnn3000_noTpl_gpu08}"

mkdir -p "$RESULTS_ROOT"
FORCE="${FORCE:-1}" SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-1}" \
  bash "$SCRIPT_DIR/launch_top10_hard_boltz_mpnn1500.sh" --tmux-8gpu
