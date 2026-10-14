#!/usr/bin/env bash
# Top9：msa500_nativeQ（500 MSA + 原生 query），无 binder 模板
#
#   FORCE=1 SKIP_GPU_WAIT=1 bash launch_top10_msa500_nativeQ_noTpl.sh --tmux-9
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/data/wenhao/Protein-Hunter-server"
SCAFFOLDS="$REPO/test_scaffolds"
SCAFFORDS_ROOT="${SCAFFORDS_ROOT:-$REPO/test_scaffords_5.20_msa500_nativeQ_noTpl}"
TAG="${TAG:-_msa500_nativeQ}"
MSA_N=500
MSA_MAX=500

# 8EO5 原生 a3m
if [[ ! -f "$SCAFFOLDS/8EO5/8EO5.a3m" && -f "$REPO/11/8EO5/8EO5.a3m" ]]; then
  ln -sfn "$REPO/11/8EO5/8EO5.a3m" "$SCAFFOLDS/8EO5/8EO5.a3m"
fi

mkdir -p "$SCAFFORDS_ROOT"
IDS="8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX"
for id in $IDS; do
  if [[ ! -f "$SCAFFOLDS/$id/${id}${TAG}.a3m" ]]; then
    echo "missing $SCAFFOLDS/$id/${id}${TAG}.a3m — run build_a3m_subset_native_query.py first" >&2
    exit 1
  fi
done

IDS="$IDS" TAG="$TAG" SRC="$SCAFFOLDS" DST="$SCAFFORDS_ROOT" \
  bash "$SCRIPT_DIR/prepare_scaffords_inputs.sh"

export USE_BINDER_TEMPLATE=0 ALLOW_QUERY_ALLX=0
export MSA_N MSA_MAX TAG INPUT_ROOT="$SCAFFORDS_ROOT" RESULTS_ROOT="$SCAFFORDS_ROOT"
export SESSION="${SESSION:-boltz_top9_msa500_nativeQ_noTpl_gpu567}"

MODE="${1:---tmux-9}"
FORCE="${FORCE:-1}" SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-1}" \
  bash "$SCRIPT_DIR/launch_top10_hard_boltz_mpnn1500.sh" "$MODE"
