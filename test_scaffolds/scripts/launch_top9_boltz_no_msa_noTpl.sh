#!/usr/bin/env bash
# Top9：无 MSA、无结构模板（unconditional empty MSA + fix_all_if_no_x）
# 对照 mpnn1000 各温度组 → results_boltz_top9_no_msa_noTpl
#
#   bash launch_top9_boltz_no_msa_noTpl.sh --tmux
#   bash launch_top9_boltz_no_msa_noTpl.sh --one 8BAU 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
if [[ -n "${GPUS:-}" ]] && [[ "${GPUS}" != *"("* ]]; then
  read -ra GPUS <<< "${GPUS}"
else
  GPUS=(${GPUS[@]:-3 4 5})
fi
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

RES="$REPO/results_boltz_top9_no_msa_noTpl"
EVAL="$REPO/results_eval_boltz_top9_no_msa_noTpl"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/boltz_top9_no_msa_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local gpu="$2"
  _gpu_lock "$gpu"
  mkdir -p "$RES/$id"
  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$RES/$id/0_protein_hunter_design" 2>/dev/null || true
  fi
  local log="$RES/${id}/boltz_run.log"
  {
    echo ""
    echo "===== $(date -Is)  id=$id  no_msa  no_template  gpu=$gpu ====="
    echo "unconditional_msa_mode=empty  use_msa=false  fix_all_if_no_x  cycles=10"
  } | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --ids "$id" \
    --results_root "$RES" \
    --eval_dir "$EVAL" \
    --mode unconditional \
    --unconditional_msa_mode empty \
    --use_msa false \
    --use_template false \
    --fix_all_if_no_x \
    --num_designs 1 \
    --num_cycles 10 \
    --gpu_id 0 \
    --skip_eval \
    --boltz_model_path "$BOLTZ" \
    --ccd_path "$CCD" \
    2>&1 | tee -a "$log"
  echo "===== done $id $(date -Is) =====" | tee -a "$log"
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local ids=("$@")
  local batch=()
  for id in "${ids[@]}"; do
    batch+=("$id")
    if [[ ${#batch[@]} -ge $PROCS_PER_GPU ]]; then
      local pids=()
      for bid in "${batch[@]}"; do
        echo ">>> no_msa gpu=$gpu start $bid"
        run_one "$bid" "$gpu" &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    local pids=()
    for bid in "${batch[@]}"; do
      echo ">>> no_msa gpu=$gpu start $bid"
      run_one "$bid" "$gpu" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
}

assign_and_run() {
  local i=0
  declare -A QUEUE
  for g in "${GPUS[@]}"; do QUEUE[$g]=""; done
  for id in "${TOP9[@]}"; do
    local g="${GPUS[$((i % ${#GPUS[@]}))]}"
    QUEUE[$g]+="$id "
    i=$((i + 1))
  done
  echo "########## Boltz Top9 no_msa no_template ##########"
  for g in "${GPUS[@]}"; do
    # shellcheck disable=SC2206
    ids=(${QUEUE[$g]})
    [[ ${#ids[@]} -eq 0 ]] && continue
    run_gpu_queue "$g" "${ids[@]}" &
  done
  wait
  echo "########## ALL DONE ##########"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-boltz_top9_no_msa_noTpl}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    _gpus="${GPUS[*]:-3 4 5}"
    _procs="${PROCS_PER_GPU:-1}"
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
GPUS=$(printf %q "$_gpus") PROCS_PER_GPU=$(printf %q "$_procs") \
bash $(printf %q "$SCRIPT_DIR/launch_top9_boltz_no_msa_noTpl.sh"); echo DONE no_msa; exec bash"
    tmux new-session -d -s "$SESSION" -n "no_msa" "$inner"
    echo "tmux: $SESSION  GPUs=${_gpus}"
    echo "  输出: $RES"
    ;;
  --one)
    run_one "${2:?id}" "${3:-3}"
    ;;
  "")
    assign_and_run
    ;;
  *)
    echo "usage: $0 [--tmux | --one ID GPU]" >&2
    exit 1
    ;;
esac
