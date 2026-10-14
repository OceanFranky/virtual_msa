#!/usr/bin/env bash
# Top9 × T=1.0,0.5,0.2,0.01 → 各 1000 条非重复 gap 对齐 MSA（高温优先）
#
#   bash launch_top9_mpnn1000_temp_sweep.sh --tmux
#   bash launch_top9_mpnn1000_temp_sweep.sh --one 8BAU 1.0 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn1000_temp_msa.sh"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
# 从高温到低温
TEMPS=(1.0 0.5 0.2 0.01)
GPUS=(${GPUS:-3 4 5 6 7 3 4 5 6})

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/mpnn1000t_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local temp="$2"
  local gpu="${3:-3}"
  _gpu_lock "$gpu"
  export CASE_DIR="$ROOT/$id" CASE_ID="$id" TEMPERATURE="$temp"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"
}

_tmux_temp_cmd() {
  local temp="$1"
  shift
  local ids=("$@")
  local parts=()
  local i=0
  for id in "${ids[@]}"; do
    local gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    parts+=("bash $(printf %q "$SCRIPT_DIR/launch_top9_mpnn1000_temp_sweep.sh") --one $(printf %q "$id") $(printf %q "$temp") $(printf %q "$gpu")")
    i=$((i + 1))
  done
  local chain
  chain=$(IFS='; '; echo "${parts[*]}")
  printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; %s; echo DONE T=%s; exec bash' "$chain" "$temp"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-mpnn1000_temp_top9}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
bash $(printf %q "$SCRIPT_DIR/launch_top9_mpnn1000_temp_sweep.sh"); echo ALL DONE; exec bash"
    tmux new-session -d -s "$SESSION" -n "sweep" "$inner"
    echo "tmux: $SESSION  温度顺序 ${TEMPS[*]}（每档 T 内 9 PDB 多卡并行）"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    run_one "${2:?id}" "${3:?temp}" "${4:-3}"
    ;;
  "")
    for temp in "${TEMPS[@]}"; do
      echo ""
      echo "########## T=$temp ##########"
      pids=()
      idx=0
      for id in "${TOP9[@]}"; do
        gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
        echo ">>> start $id T=$temp gpu=$gpu"
        run_one "$id" "$temp" "$gpu" &
        pids+=($!)
        idx=$((idx + 1))
      done
      for pid in "${pids[@]}"; do
        wait "$pid" || { echo "FAIL pid=$pid" >&2; exit 1; }
      done
      echo "########## T=$temp done ##########"
    done
    ;;
  *)
    echo "usage: $0 [--tmux | --one ID TEMP GPU]" >&2
    exit 1
    ;;
esac
