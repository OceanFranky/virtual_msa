#!/usr/bin/env bash
# Top9：原生 {ID}.a3m（~500 MSA）query 全 X → boltz，**无 binder 结构模板**
# （此前 _250/_1000/_1500 均带 mask.pdb 模板；本批仅 MSA + 全 X 序列）
#
#   bash launch_top10_native500_queryx.sh --prep-only
#   FORCE=1 SKIP_GPU_WAIT=1 bash launch_top10_native500_queryx.sh --tmux-9
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/data/wenhao/Protein-Hunter-server"
SCAFFOLDS="$REPO/test_scaffolds"
SCAFFORDS_ROOT="${SCAFFORDS_ROOT:-$REPO/test_scaffords_5.20_native500_queryX_noTpl}"
TAG="${TAG:-_native500_queryX}"
MSA_MAX="${MSA_MAX:-500}"
PYTHON="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)

prep_msa() {
  # 8EO5 无 test_scaffolds/{ID}.a3m 时从 11/ 取
  if [[ ! -f "$SCAFFOLDS/8EO5/8EO5.a3m" && -f "$REPO/11/8EO5/8EO5.a3m" ]]; then
    ln -sfn "$REPO/11/8EO5/8EO5.a3m" "$SCAFFOLDS/8EO5/8EO5.a3m"
    echo "linked 8EO5.a3m from 11/"
  fi
  "$PYTHON" "$SCRIPT_DIR/make_a3m_query_allx.py" \
    --ids "${TOP9[@]}" \
    --src-suffix "" \
    --out-suffix "$TAG" \
    --root "$SCAFFOLDS"
}

prep_inputs() {
  prep_msa
  IDS="${TOP9[*]}" TAG="$TAG" SRC="$SCAFFOLDS" DST="$SCAFFORDS_ROOT" \
    bash "$SCRIPT_DIR/prepare_scaffords_inputs.sh"
  echo "inputs ready: $SCAFFORDS_ROOT  TAG=$TAG  MSA_MAX=$MSA_MAX"
}

MODE="${1:-}"
case "$MODE" in
  --prep-only)
    prep_inputs
    ;;
  --tmux-9|"")
    prep_inputs
    export ALLOW_QUERY_ALLX=1 USE_BINDER_TEMPLATE=0
    export MSA_N=500 MSA_MAX="$MSA_MAX" TAG="$TAG"
    export INPUT_ROOT="$SCAFFORDS_ROOT" RESULTS_ROOT="$SCAFFORDS_ROOT"
    export SESSION="${SESSION:-boltz_top9_native500_queryX_noTpl_gpu567}"
    FORCE="${FORCE:-1}" SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-1}" \
      bash "$SCRIPT_DIR/launch_top10_hard_boltz_mpnn1500.sh" --tmux-9
    ;;
  *)
    echo "usage: $0 [--prep-only|--tmux-9]" >&2
    exit 1
    ;;
esac
