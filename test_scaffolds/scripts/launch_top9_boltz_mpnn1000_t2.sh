#!/usr/bin/env bash
# Top9 × mpnn1000_t2.0 MSA：无 template，序列固定（同 t0.01）
#
#   bash launch_top9_boltz_mpnn1000_t2.sh --tmux
#   bash launch_top9_boltz_mpnn1000_t2.sh --one 8BAU 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"
TAG="${TAG:-t2.0}"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
# 支持 GPUS="3 4 5" 或已 export 的数组；默认仅用 3/4/5 三卡
if [[ -n "${GPUS:-}" ]] && [[ "${GPUS}" != *"("* ]]; then
  read -ra GPUS <<< "${GPUS}"
else
  GPUS=(${GPUS[@]:-3 4 5})
fi
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/boltz_top9_t2_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local gpu="$2"
  _gpu_lock "$gpu"
  local res="$REPO/results_boltz_top9_mpnn1000_${TAG}"
  local eval="$REPO/results_eval_boltz_top9_mpnn1000_${TAG}"
  local msa_suffix="_mpnn1000_${TAG}"
  local a3m="$ROOT/$id/${id}${msa_suffix}.a3m"
  if [[ ! -f "$a3m" ]]; then
    echo "missing MSA: $a3m" >&2
    exit 1
  fi
  mkdir -p "$res/$id"
  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$res/$id/0_protein_hunter_design" 2>/dev/null || true
  fi
  local log="$res/${id}/boltz_run.log"
  {
    echo ""
    echo "===== $(date -Is)  id=$id  msa=${msa_suffix}  gpu=$gpu ====="
    echo "no_template  fix_all_if_no_x  num_designs=1  num_cycles=10  msa_max_seqs=1000"
  } | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --ids "$id" \
    --msa_tag "$msa_suffix" \
    --msa_max_seqs 1000 \
    --results_root "$res" \
    --eval_dir "$eval" \
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
    2>&1 | tee -a "$log"
  echo "===== done $id ${TAG} $(date -Is) =====" | tee -a "$log"
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
        echo ">>> ${TAG} gpu=$gpu start $bid"
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
      echo ">>> ${TAG} gpu=$gpu start $bid"
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
  echo "########## Boltz mpnn1000_${TAG} ##########"
  for g in "${GPUS[@]}"; do
    # shellcheck disable=SC2206
    ids=(${QUEUE[$g]})
    [[ ${#ids[@]} -eq 0 ]] && continue
    run_gpu_queue "$g" "${ids[@]}" &
  done
  wait
  echo "########## ${TAG} done ##########"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-boltz_top9_mpnn1000_t2}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    _gpus="${GPUS[*]:-3 4 5}"
    _procs="${PROCS_PER_GPU:-1}"
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
GPUS=$(printf %q "$_gpus") PROCS_PER_GPU=$(printf %q "$_procs") \
bash $(printf %q "$SCRIPT_DIR/launch_top9_boltz_mpnn1000_t2.sh"); echo DONE boltz t2.0; exec bash"
    tmux new-session -d -s "$SESSION" -n "t2.0" "$inner"
    echo "tmux: $SESSION"
    echo "  输出: results_boltz_top9_mpnn1000_t2.0/"
    echo "  评测: results_eval_boltz_top9_mpnn1000_t2.0/ (跑 eval_top9_mpnn1000_t2.py)"
    ;;
  --one)
    run_one "${2:?id}" "${3:-2}"
    ;;
  "")
    assign_and_run
    ;;
  *)
    echo "usage: $0 [--tmux | --one ID GPU]" >&2
    exit 1
    ;;
esac
