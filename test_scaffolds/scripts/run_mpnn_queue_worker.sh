#!/usr/bin/env bash
# 从 mpnn_batch_pending.tsv 按 worker 分片跑 MPNN 3000 + 切分 250/1000/1500/2000
#   GPU=0 WORKERS=8 bash run_mpnn_queue_worker.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build_mpnn_msa_unique.sh"
SPLIT="$SCRIPT_DIR/split_mpnn_a3m_by_score.py"
QUEUE="${QUEUE:-$SCRIPT_DIR/mpnn_batch_pending.tsv}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
TARGET_SEQS="${TARGET_SEQS:-3000}"
TOP_SPLIT="${TOP_SPLIT:-250 500 1000 1500 2000}"

set +u
source /data/wenhao/.bashrc 2>/dev/null || true
conda activate proteinhunter 2>/dev/null || true
set -u
export PATH="/data/wenhao/miniconda3/envs/proteinhunter/bin:${PATH:-}"
PYTHON="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"

line_no=0
while IFS=$'\t' read -r cid root || [[ -n "${cid:-}" ]]; do
  [[ "$cid" == "id" || -z "${cid:-}" ]] && continue
  if (( line_no % WORKERS != GPU )); then
    line_no=$((line_no + 1))
    continue
  fi
  line_no=$((line_no + 1))
  echo ">>> [GPU$GPU] $cid  root=$root"
  if [[ -f "$root/$cid/${cid}_mpnn3000_unique.npz" ]]; then
    echo "    skip (已有 mpnn3000)"
    continue
  fi
  export CASE_DIR="$root/$cid" CASE_ID="$cid" TARGET_SEQS
  export OUT_TAG="_mpnn${TARGET_SEQS}_unique"
  export WORK="$CASE_DIR/mpnn_work_${TARGET_SEQS}_unique"
  if CUDA_VISIBLE_DEVICES="$GPU" bash "$BUILD"; then
    "$PYTHON" "$SPLIT" --root "$root" --ids "$cid" --top $TOP_SPLIT --no-csv
    echo ">>> DONE $cid"
  else
    echo ">>> FAIL $cid (see ${cid}_mpnn3000_unique.log)" >&2
  fi
done < "$QUEUE"

echo "worker GPU=$GPU finished $(date -Is)"
