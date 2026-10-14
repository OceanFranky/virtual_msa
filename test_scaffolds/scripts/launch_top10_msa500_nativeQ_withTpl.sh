#!/usr/bin/env bash
# Top9：msa500_nativeQ（500 MSA + 原生 query）+ mask.pdb 模板（force=False）
# 对照 msa500_nativeQ_noTpl → test_scaffords_5.20_msa500_nativeQ_withTpl
#
#   FORCE=1 SKIP_GPU_WAIT=1 bash launch_top10_msa500_nativeQ_withTpl.sh --tmux-9
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/data/wenhao/Protein-Hunter-server"
SCAFFOLDS="$REPO/test_scaffolds"
SCAFFORDS_ROOT="${SCAFFORDS_ROOT:-$REPO/test_scaffords_5.20_msa500_nativeQ_withTpl}"
TAG="${TAG:-_msa500_nativeQ}"
MSA_N=500
MSA_MAX=500

if [[ ! -f "$SCAFFOLDS/8EO5/8EO5.a3m" && -f "$REPO/11/8EO5/8EO5.a3m" ]]; then
  ln -sfn "$REPO/11/8EO5/8EO5.a3m" "$SCAFFOLDS/8EO5/8EO5.a3m"
fi

IDS="8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX"
for id in $IDS; do
  [[ -f "$SCAFFOLDS/$id/${id}${TAG}.a3m" ]] || {
    echo "missing $SCAFFOLDS/$id/${id}${TAG}.a3m" >&2; exit 1; }
done

mkdir -p "$SCAFFORDS_ROOT"
IDS="$IDS" TAG="$TAG" SRC="$SCAFFOLDS" DST="$SCAFFORDS_ROOT" \
  bash "$SCRIPT_DIR/prepare_scaffords_inputs.sh"

export USE_BINDER_TEMPLATE=1 ALLOW_QUERY_ALLX=0
export MSA_N MSA_MAX TAG INPUT_ROOT="$SCAFFORDS_ROOT" RESULTS_ROOT="$SCAFFORDS_ROOT"
export SESSION="${SESSION:-boltz_top9_msa500_nativeQ_withTpl_gpu567}"

FORCE="${FORCE:-1}" SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-1}" \
  bash "$SCRIPT_DIR/launch_top10_hard_boltz_mpnn1500.sh" "${1:---tmux-9}"
