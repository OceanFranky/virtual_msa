#!/usr/bin/env bash
# 批量 MPNN → 3000 条不重复 a3m/npz
#
#   IDS="9FGP 9FOW 9FYM 9GSO 9IJC 9IUH 9J0V 9JQ9" bash launch_scaffold_series_mpnn3000.sh --tmux
#   FORCE=1 可重开 tmux
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build_mpnn_msa_unique.sh"
ROOT="${ROOT:-/data/wenhao/Protein-Hunter-server/test_scaffolds}"
TARGET_SEQS="${TARGET_SEQS:-3000}"
SESSION="${SESSION:-mpnn_scaffold_3k}"

# 5 蛋白 ↔ GPU 0–4；不跑 9IUH 9J0V 9JQ9
DEFAULT_IDS=(9FGP 9FOW 9FYM 9GSO 9IJC)
if [[ -n "${IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ID_ARR=($IDS)
else
  ID_ARR=("${DEFAULT_IDS[@]}")
fi
NGPU="${NGPU:-16}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"

run_one() {
  local id="$1" gpu="${2:-0}"
  export CASE_DIR="$ROOT/$id" CASE_ID="$id" TARGET_SEQS
  export OUT_TAG="_mpnn${TARGET_SEQS}_unique"
  export WORK="$CASE_DIR/mpnn_work_${TARGET_SEQS}_unique"
  echo "===== MPNN $id GPU=$gpu ====="
  CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"
}

MODE="${1:-}"
case "$MODE" in
  --tmux)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    n=${#ID_ARR[@]}
    if [[ "$n" -gt "$MAX_PARALLEL" ]]; then
      echo "错误: ${n} 个蛋白 > MAX_PARALLEL=$MAX_PARALLEL" >&2
      exit 1
    fi
    i=0
    for id in "${ID_ARR[@]}"; do
      gpu=$((i % NGPU))
      cmd="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
export PATH=/data/wenhao/miniconda3/envs/proteinhunter/bin:\$PATH; \
CASE_DIR=$(printf %q "$ROOT/$id") CASE_ID=$(printf %q "$id") TARGET_SEQS=$TARGET_SEQS \
CUDA_VISIBLE_DEVICES=$gpu bash $(printf %q "$BUILD"); \
echo DONE MPNN $id; exec bash"
      if [[ "$i" -eq 0 ]]; then
        tmux new-session -d -s "$SESSION" -n "$id" "$cmd"
      else
        tmux new-window -t "$SESSION" -n "$id" "$cmd"
      fi
      i=$((i + 1))
    done
    echo "tmux: $SESSION  (${#ID_ARR[@]} windows, GPU=i%${NGPU}, i=0..$((n-1)))"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    run_one "${2:?id}" "${3:-0}"
    ;;
  "")
    i=0
    for id in "${ID_ARR[@]}"; do
      run_one "$id" $((i % NGPU))
      i=$((i + 1))
    done
    ;;
  *)
    echo "usage: IDS='...' $0 [--tmux|--one ID GPU]" >&2
    exit 1
    ;;
esac
