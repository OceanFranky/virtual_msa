#!/usr/bin/env bash
# 8G52 → 8G53 → 8G64 → 8GAC → 8GBE：各 3000 条不重复 MPNN MSA
#
#   bash launch_8G52_series_mpnn3000.sh          # 前台串行（稳）
#   bash launch_8G52_series_mpnn3000.sh --parallel  # 5 路并行
#   bash launch_8G52_series_mpnn3000.sh --tmux       # tmux 并行
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build_mpnn_msa_unique.sh"
ROOT="${ROOT:-/data/wenhao/Protein-Hunter-server/test_scaffolds}"
TARGET_SEQS="${TARGET_SEQS:-3000}"

IDS=(8G52 8G53 8G64 8GAC 8GBE)

run_one() {
  local id="$1"
  export CASE_DIR="$ROOT/$id"
  export CASE_ID="$id"
  export TARGET_SEQS
  export OUT_TAG="_mpnn${TARGET_SEQS}_unique"
  export WORK="$CASE_DIR/mpnn_work_${TARGET_SEQS}_unique"
  bash "$BUILD"
}

MODE="${1:-}"
case "$MODE" in
  --parallel)
    pids=()
    for id in "${IDS[@]}"; do
      run_one "$id" &
      pids+=($!)
    done
    fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    exit "$fail"
    ;;
  --tmux)
    SESSION="${SESSION:-mpnn_8G_series_3k}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux 已存在: $SESSION — tmux attach -t $SESSION" >&2
      exit 1
    fi
    tmux new-session -d -s "$SESSION" -n "8G52" \
      "CASE_DIR=$(printf %q "$ROOT/8G52") CASE_ID=8G52 TARGET_SEQS=$TARGET_SEQS bash $(printf %q "$BUILD"); exec bash"
    for id in 8G53 8G64 8GAC 8GBE; do
      tmux new-window -t "$SESSION" -n "$id" \
        "CASE_DIR=$(printf %q "$ROOT/$id") CASE_ID=$id TARGET_SEQS=$TARGET_SEQS bash $(printf %q "$BUILD"); exec bash"
    done
    echo "tmux: $SESSION  (5 windows)"
    exit 0
    ;;
  "")
    for id in "${IDS[@]}"; do
      echo "######## $id ########"
      run_one "$id"
    done
    ;;
  *)
    echo "usage: $0 [--parallel|--tmux]" >&2
    exit 1
    ;;
esac

echo "all done: ${IDS[*]}"
