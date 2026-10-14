#!/usr/bin/env bash
# 批量重建 11/ + test_scaffolds 全部 gap 对齐 a3m
#   bash launch_rebuild_a3m_gap_aligned.sh
#   bash launch_rebuild_a3m_gap_aligned.sh --tmux
#
set -euo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/rebuild_a3m_gap_aligned.py"
PYTHON="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
LOG="/data/wenhao/Protein-Hunter-server/test_scaffolds/scripts/rebuild_a3m_gap_aligned.log"

if [[ "${1:-}" == "--tmux" ]]; then
  SESSION="${SESSION:-rebuild_a3m_gap_x8}"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" \
    "set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
     $PYTHON $SCRIPT 2>&1 | tee -a $LOG; echo DONE; exec bash"
  echo "tmux: $SESSION  log: $LOG"
  echo "attach: tmux attach -t $SESSION"
else
  "$PYTHON" "$SCRIPT" 2>&1 | tee -a "$LOG"
fi
