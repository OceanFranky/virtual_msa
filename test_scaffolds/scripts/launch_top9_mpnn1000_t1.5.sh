#!/usr/bin/env bash
# Top9 × MPNN T=1.5 → 各 1000 条非重复 gap 对齐 MSA
#
#   bash launch_top9_mpnn1000_t1.5.sh --tmux
#   GPUS="0 1 2 3 4 5 6 7" bash launch_top9_mpnn1000_t1.5.sh
#   bash launch_top9_mpnn1000_t1.5.sh --one 8BAU 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn1000_temp_msa.sh"
TEMPERATURE="${TEMPERATURE:-1.5}"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
if [[ -n "${GPUS:-}" ]] && [[ "${GPUS}" != *"("* ]]; then
  read -ra GPUS <<< "${GPUS}"
else
  GPUS=(${GPUS[@]:-0 1 2 3 4 5 6 7})
fi

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/mpnn1000t15_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local gpu="${2:-0}"
  _gpu_lock "$gpu"
  export CASE_DIR="$ROOT/$id" CASE_ID="$id" TEMPERATURE="$TEMPERATURE"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-mpnn1000_t1.5_top9}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    _gpus="${GPUS[*]:-0 1 2 3 4 5 6 7}"
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
GPUS=$(printf %q "$_gpus") bash $(printf %q "$SCRIPT_DIR/launch_top9_mpnn1000_t1.5.sh"); echo ALL DONE T=1.5; exec bash"
    tmux new-session -d -s "$SESSION" -n "t1.5" "$inner"
    echo "tmux: $SESSION  T=$TEMPERATURE  GPUs=${_gpus}  (9 PDB 并行)"
    echo "  输出: test_scaffolds/{ID}/{ID}_mpnn1000_t1.5.a3m"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    run_one "${2:?id}" "${3:-0}"
    ;;
  "")
    pids=()
    idx=0
    for id in "${TOP9[@]}"; do
      gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      echo ">>> start $id T=$TEMPERATURE gpu=$gpu"
      run_one "$id" "$gpu" &
      pids+=($!)
      idx=$((idx + 1))
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || { echo "FAIL pid=$pid" >&2; exit 1; }
    done
    echo "ALL DONE mpnn1000 T=$TEMPERATURE"
    ;;
  *)
    echo "usage: $0 [--tmux | --one ID GPU]" >&2
    exit 1
    ;;
esac
