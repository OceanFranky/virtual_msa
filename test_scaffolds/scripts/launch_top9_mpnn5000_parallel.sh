#!/usr/bin/env bash
# Top9 × mpnn5000_unique：多 GPU，每卡最多 2 个 boltz 进程并行
#
#   bash launch_top9_mpnn5000_parallel.sh --tmux
#   GPUS="3 4 5 6 7" bash launch_top9_mpnn5000_parallel.sh --tmux
#   bash launch_top9_mpnn5000_parallel.sh --one 8U0X 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"
MSA_N=5000
RES="$REPO/results_boltz_top9_mpnn${MSA_N}"
EVAL="$REPO/results_eval_boltz_top9_mpnn${MSA_N}"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
# 默认用较空的卡；可通过环境变量 GPUS 覆盖
GPUS=(${GPUS:-3 4 5 6 7})
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"

run_one() {
  local id="$1"
  local gpu="$2"
  local log="$RES/${id}/boltz_run.log"
  mkdir -p "$RES/$id"
  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$RES/$id/0_protein_hunter_design" 2>/dev/null || true
  fi
  {
    echo ""
    echo "===== $(date -Is)  id=$id  msa_n=$MSA_N  gpu=$gpu  (CUDA_VISIBLE_DEVICES=$gpu) ====="
    echo "fix_all_if_no_x  no_template  num_designs=1  num_cycles=10"
  } | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --ids "$id" \
    --msa_n "$MSA_N" \
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
    2>&1 | tee -a "$log"
  echo "===== done $id $(date -Is) =====" | tee -a "$log"
}

# 同一 GPU 上并行跑最多 PROCS_PER_GPU 个，多出来的 ID 在该卡上串行排队
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
        echo ">>> gpu=$gpu  start $bid (parallel)"
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
      echo ">>> gpu=$gpu  start $bid"
      run_one "$bid" "$gpu" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
}

# 把 TOP9 均分到各 GPU
assign_queues() {
  local n_gpus=${#GPUS[@]}
  local i=0
  for id in "${TOP9[@]}"; do
    local g="${GPUS[$((i % n_gpus))]}"
    eval "QUEUE_GPU${g}+=(\"$id\")"
    i=$((i + 1))
  done
}

_tmux_gpu_cmd() {
  local gpu="$1"
  shift
  local ids=("$@")
  local inner_ids
  inner_ids=$(printf '%q ' "${ids[@]}")
  printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
export REPO=%q ROOT=%q GPUS=%q PROCS_PER_GPU=%q CLEAN_PARTIAL=%q; \
bash %q --gpu-queue %s %s; echo DONE gpu%s; exec bash' \
    "$REPO" "$ROOT" "$gpu" "$PROCS_PER_GPU" "${CLEAN_PARTIAL:-1}" \
    "$SCRIPT_DIR/launch_top9_mpnn5000_parallel.sh" "$gpu" "$inner_ids" "$gpu"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-boltz_top9_mpnn5000_par}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    mkdir -p "$RES"
    assign_queues
    first=1
    for gpu in "${GPUS[@]}"; do
      eval "ids=(\"\${QUEUE_GPU${gpu}[@]-}\")"
      [[ ${#ids[@]} -eq 0 ]] && continue
      if [[ $first -eq 1 ]]; then
        tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$(_tmux_gpu_cmd "$gpu" "${ids[@]}")"
        first=0
      else
        tmux new-window -t "$SESSION" -n "gpu${gpu}" "$(_tmux_gpu_cmd "$gpu" "${ids[@]}")"
      fi
    done
    echo "tmux: $SESSION  out=$RES  MSA=$MSA_N  procs_per_gpu=$PROCS_PER_GPU"
    for gpu in "${GPUS[@]}"; do
      eval "echo \"  GPU${gpu}: \${QUEUE_GPU${gpu}[*]-}\""
    done
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --gpu-queue)
    run_gpu_queue "${2:?gpu}" "${@:3}"
    ;;
  --one)
    mkdir -p "$RES"
    run_one "${2:?id}" "${3:?gpu}"
    ;;
  --stop-old)
    tmux kill-session -t boltz_top9_mpnn5000_gpu2 2>/dev/null || true
    pkill -f "results_boltz_top9_mpnn5000" 2>/dev/null || true
    sleep 2
    echo "已停止旧 mpnn5000 单卡任务"
    ;;
  "")
    echo "usage: $0 [--tmux | --stop-old | --one ID GPU | --gpu-queue GPU ID ...]" >&2
    exit 1
    ;;
  *)
    echo "unknown mode: $1" >&2
    exit 1
    ;;
esac
