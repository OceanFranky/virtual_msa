#!/usr/bin/env bash
# 全量 test_scaffolds：mpnn250 / mpnn2000，无 template，GPU 6 / 7 各跑一组
#   bash launch_scaffolds_mpnn250_2000_gpu67.sh 250 6
#   bash launch_scaffolds_mpnn250_2000_gpu67.sh 2000 7
#   bash launch_scaffolds_mpnn250_2000_gpu67.sh --tmux   # 同时开两个 tmux 窗
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"

run_msa_n() {
  local N="$1"
  local GPU="$2"
  local RES="$REPO/results_boltz_mpnn${N}"
  local EVAL="$REPO/results_eval_boltz_mpnn${N}"
  local LOG="$RES/batch_run.log"

  mkdir -p "$RES"
  if [[ "${CLEAN_PREVIOUS:-1}" == "1" ]]; then
    find "$RES" -mindepth 2 -maxdepth 2 -type d -name '0_protein_hunter_design' -prune -exec rm -rf {} + 2>/dev/null || true
    echo "cleared old 0_protein_hunter_design under $RES (CLEAN_PREVIOUS=1)"
  fi
  {
    echo ""
    echo "===== RESTART $(date -Is)  msa_n=$N  gpu=$GPU  root=$ROOT ====="
    echo "results=$RES  eval=$EVAL"
    echo "use_template=false  --fix_all_if_no_x  num_designs=1  num_cycles=5"
  } | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --msa_n "$N" \
    --results_root "$RES" \
    --eval_dir "$EVAL" \
    --mode unconditional \
    --unconditional_msa_mode precomputed \
    --use_msa true \
    --use_template false \
    --fix_all_if_no_x \
    --num_designs 1 \
    --num_cycles 5 \
    --gpu_id 0 \
    --skip_eval \
    --boltz_model_path "$BOLTZ" \
    --ccd_path "$CCD" \
    2>&1 | tee -a "$LOG"

  echo "===== done $(date -Is)  msa_n=$N =====" | tee -a "$LOG"
}

_tmux_one() {
  local N="$1"
  local GPU="$2"
  local SESS="boltz_scaffolds_mpnn${N}_gpu${GPU}"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "tmux 已存在: $SESS（如需重跑: tmux kill-session -t $SESS）" >&2
    return 1
  fi
  local cmd
  cmd=$(printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; bash %q %s %s 2>&1 | tee -a %q; echo DONE msa_n=%s; exec bash' \
    "$SCRIPT_DIR/launch_scaffolds_mpnn250_2000_gpu67.sh" "$N" "$GPU" \
    "$REPO/results_boltz_mpnn${N}/batch_run.log" "$N")
  tmux new-session -d -s "$SESS" -n "mpnn${N}" "$cmd"
  echo "tmux: $SESS  GPU=$GPU  log=$REPO/results_boltz_mpnn${N}/batch_run.log"
  echo "  attach: tmux attach -t $SESS"
}

case "${1:-}" in
  --tmux)
    _tmux_one 250 6
    _tmux_one 2000 7
    ;;
  "")
    echo "usage: $0 <250|2000> <gpu_id>   OR   $0 --tmux" >&2
    exit 1
    ;;
  *)
    run_msa_n "$1" "${2:?gpu_id}"
    ;;
esac
