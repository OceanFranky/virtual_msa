#!/usr/bin/env bash
# Top9：T=0.1→2.0 步长 0.1，每档 ~50 条，合并为单个 a3m（保留，不删除）
#
#   bash launch_top9_mpnn_tstep_single_a3m.sh
#   bash launch_top9_mpnn_tstep_single_a3m.sh --one 8BAU 0
#   bash launch_top9_mpnn_tstep_single_a3m.sh --tmux
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn_tstep_single_a3m.sh"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
GPUS=(${GPUS:-0 6 2 4})
OUT_TAG="${OUT_TAG:-_mpnn50_t01_20}"

_gpu_lock() {
  local gpu="$1"
  exec 9>"/tmp/mpnn_tstep_top9_gpu${gpu}.lock"
  flock 9
}

run_one() {
  local id="$1"
  local gpu="${2:-0}"
  _gpu_lock "$gpu"
  export CASE_DIR="$ROOT/$id" CASE_ID="$id" OUT_TAG="$OUT_TAG" KEEP_INTERMEDIATE=1
  CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"
}

case "${1:-}" in
  --one)
    run_one "${2:?id}" "${3:-0}"
    ;;
  --tmux)
    SESSION="${SESSION:-mpnn_tstep_top9}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
bash $(printf %q "$SCRIPT_DIR/launch_top9_mpnn_tstep_single_a3m.sh"); echo ALL DONE; exec bash"
    tmux new-session -d -s "$SESSION" -n sweep "$inner"
    echo "tmux: $SESSION  attach: tmux attach -t $SESSION"
    echo "  输出: test_scaffolds/{ID}/{ID}${OUT_TAG}.a3m"
    ;;
  "")
    pids=() idx=0
    for id in "${TOP9[@]}"; do
      gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      echo ">>> start $id gpu=$gpu"
      run_one "$id" "$gpu" &
      pids+=($!)
      idx=$((idx + 1))
    done
    fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    [[ "$fail" -eq 0 ]] || { echo "FAIL: some jobs failed" >&2; exit 1; }
    echo ""
    echo "ALL DONE Top9 mpnn tstep 0.1-2.0 -> ${OUT_TAG}.a3m (kept)"
    for id in "${TOP9[@]}"; do
      f="$ROOT/$id/${id}${OUT_TAG}.a3m"
      if [[ -f "$f" ]]; then
        n=$(grep -c '^>' "$f" || true)
        echo "  $id  headers=$n  $f"
      else
        echo "  $id  MISSING $f" >&2
      fi
    done
    ;;
  *)
    echo "usage: $0 [--one ID GPU | --tmux]" >&2
    exit 1
    ;;
esac
