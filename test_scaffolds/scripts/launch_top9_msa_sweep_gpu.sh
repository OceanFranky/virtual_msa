#!/usr/bin/env bash
# Top9 PDB × MSA 250/1500/2000/3000/5000：无 template，a3m 第一条序列 + fix_all_if_no_x
#   1 run × 10 cycles；GPU0 串行 250→1500；GPU2/3/4 分别 1000/2000/3000
#
#   bash launch_top9_msa_sweep_gpu.sh --tmux
#   bash launch_top9_msa_sweep_gpu.sh 250 0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
IDS_STR="${TOP9[*]}"

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
  local RES="$REPO/results_boltz_top9_mpnn${N}"
  local EVAL="$REPO/results_eval_boltz_top9_mpnn${N}"
  local LOG="$RES/batch_run.log"

  mkdir -p "$RES"
  if [[ "${CLEAN_PREVIOUS:-1}" == "1" ]]; then
    find "$RES" -mindepth 2 -maxdepth 2 -type d -name '0_protein_hunter_design' -prune -exec rm -rf {} + 2>/dev/null || true
  fi
  {
    echo ""
    echo "===== $(date -Is)  top9 msa_n=$N  gpu=$GPU  ids=$IDS_STR ====="
    echo "fix_all_if_no_x  no_template  num_designs=1  num_cycles=10"
  } | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --ids "${TOP9[@]}" \
    --msa_n "$N" \
    --results_root "$RES" \
    --eval_dir "$EVAL" \
    --mode unconditional \
    --unconditional_msa_mode precomputed \
    --use_msa true \
    --use_template false \
    --fix_all_if_no_x \
    --num_designs 1 \
    --num_cycles 10 \
    --gpu_id 0 \
    --skip_eval \
    --boltz_model_path "$BOLTZ" \
    --ccd_path "$CCD" \
    2>&1 | tee -a "$LOG"

  echo "===== done msa_n=$N $(date -Is) =====" | tee -a "$LOG"
}

run_gpu0_serial() {
  run_msa_n 250 0
  run_msa_n 1500 0
}

_tmux_one() {
  local name="$1"
  local gpu="$2"
  shift 2
  local SESS="boltz_top9_${name}"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "tmux 已存在: $SESS — kill 后重试" >&2
    return 1
  fi
  local inner
  inner=$(printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; %s; echo DONE %s; exec bash' "$*" "$name")
  tmux new-session -d -s "$SESS" -n "$name" "$inner"
  echo "tmux: $SESS  GPU=$gpu"
  echo "  attach: tmux attach -t $SESS"
}

case "${1:-}" in
  --tmux)
    _tmux_one mpnn250_1500_gpu0 0 \
      "bash $(printf %q "$SCRIPT_DIR/launch_top9_msa_sweep_gpu.sh") --gpu0-serial"
    _tmux_one mpnn1000_gpu2 2 \
      "bash $(printf %q "$SCRIPT_DIR/launch_top9_msa_sweep_gpu.sh") 1000 2"
    _tmux_one mpnn2000_gpu3 3 \
      "bash $(printf %q "$SCRIPT_DIR/launch_top9_msa_sweep_gpu.sh") 2000 3"
    _tmux_one mpnn3000_gpu4 4 \
      "bash $(printf %q "$SCRIPT_DIR/launch_top9_msa_sweep_gpu.sh") 3000 4"
    ;;
  --gpu0-serial)
    run_gpu0_serial
    ;;
  "")
    echo "usage: $0 --tmux | --gpu0-serial | <msa_n> <gpu>" >&2
    exit 1
    ;;
  *)
    run_msa_n "$1" "${2:?gpu}"
    ;;
esac
