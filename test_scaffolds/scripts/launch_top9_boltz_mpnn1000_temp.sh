#!/usr/bin/env bash
# Top9 × mpnn1000 温度 MSA：无 template，序列固定，GPU 2/3/4/5/0 每卡 2 进程
#
#   bash launch_top9_boltz_mpnn1000_temp.sh --tmux
#   bash launch_top9_boltz_mpnn1000_temp.sh --one 8BAU t0.5 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
# 四组温度 MSA（高温 → 低温，与 MPNN 生成顺序一致）
TEMP_TAGS=(t1.0 t0.5 t0.2 t0.01)
GPUS=(${GPUS:-2 3 4 5 0})
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/boltz_top9_t_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local tag="$2"
  local gpu="$3"
  _gpu_lock "$gpu"
  local res="$REPO/results_boltz_top9_mpnn1000_${tag}"
  local eval="$REPO/results_eval_boltz_top9_mpnn1000_${tag}"
  local msa_suffix="_mpnn1000_${tag}"
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
  echo "===== done $id ${tag} $(date -Is) =====" | tee -a "$log"
}

run_gpu_queue() {
  local tag="$1"
  local gpu="$2"
  shift 2
  local ids=("$@")
  local batch=()
  for id in "${ids[@]}"; do
    batch+=("$id")
    if [[ ${#batch[@]} -ge $PROCS_PER_GPU ]]; then
      local pids=()
      for bid in "${batch[@]}"; do
        echo ">>> ${tag} gpu=$gpu start $bid"
        run_one "$bid" "$tag" "$gpu" &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    local pids=()
    for bid in "${batch[@]}"; do
      echo ">>> ${tag} gpu=$gpu start $bid"
      run_one "$bid" "$tag" "$gpu" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
}

assign_and_run_temp() {
  local tag="$1"
  local i=0
  declare -A QUEUE
  for g in "${GPUS[@]}"; do QUEUE[$g]=""; done
  for id in "${TOP9[@]}"; do
    local g="${GPUS[$((i % ${#GPUS[@]}))]}"
    QUEUE[$g]+="$id "
    i=$((i + 1))
  done
  echo ""
  echo "########## Boltz mpnn1000_${tag} ##########"
  for g in "${GPUS[@]}"; do
    # shellcheck disable=SC2206
    ids=(${QUEUE[$g]})
    [[ ${#ids[@]} -eq 0 ]] && continue
    run_gpu_queue "$tag" "$g" "${ids[@]}" &
  done
  wait
  echo "########## ${tag} done ##########"
}

_tmux_temp_cmd() {
  local tag="$1"
  printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
bash %q --run-temp %q; echo DONE %q; exec bash' \
    "$SCRIPT_DIR/launch_top9_boltz_mpnn1000_temp.sh" "$tag" "$tag"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-boltz_top9_mpnn1000_temp}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    first=1
    for tag in "${TEMP_TAGS[@]}"; do
      if [[ $first -eq 1 ]]; then
        tmux new-session -d -s "$SESSION" -n "$tag" "$(_tmux_temp_cmd "$tag")"
        first=0
      else
        tmux new-window -t "$SESSION" -n "$tag" "$(_tmux_temp_cmd "$tag")"
      fi
    done
    echo "tmux: $SESSION  GPUs=${GPUS[*]}  procs_per_gpu=$PROCS_PER_GPU"
    echo "  4 窗并行: ${TEMP_TAGS[*]}"
    echo "  输出: results_boltz_top9_mpnn1000_{t1.0,t0.5,t0.2,t0.01}/"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --run-temp)
    assign_and_run_temp "${2:?tag}"
    ;;
  --one)
    run_one "${2:?id}" "${3:?tag}" "${4:-2}"
    ;;
  "")
    for tag in "${TEMP_TAGS[@]}"; do
      assign_and_run_temp "$tag"
    done
    echo "ALL DONE boltz mpnn1000 temp sweep"
    ;;
  *)
    echo "usage: $0 [--tmux | --run-temp TAG | --one ID TAG GPU]" >&2
    exit 1
    ;;
esac
