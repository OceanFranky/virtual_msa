#!/usr/bin/env bash
# T=2.0：先 Top9 MPNN MSA，再 Boltz + 提示评测命令
#
#   bash launch_top9_mpnn1000_t2_then_boltz.sh --tmux
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-top9_t2_full}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
bash $(printf %q "$SCRIPT_DIR/launch_top9_mpnn1000_t2.sh") && \
bash $(printf %q "$SCRIPT_DIR/launch_top9_boltz_mpnn1000_t2.sh") && \
/data/wenhao/miniconda3/envs/proteinhunter/bin/python $(printf %q "$SCRIPT_DIR/eval_top9_mpnn1000_t2.py") && \
echo ALL PIPELINE DONE; exec bash"
    tmux new-session -d -s "$SESSION" -n "full" "$inner"
    echo "tmux: $SESSION  (MSA → Boltz → eval)"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  "")
    bash "$SCRIPT_DIR/launch_top9_mpnn1000_t2.sh"
    bash "$SCRIPT_DIR/launch_top9_boltz_mpnn1000_t2.sh"
    /data/wenhao/miniconda3/envs/proteinhunter/bin/python \
      "$SCRIPT_DIR/eval_top9_mpnn1000_t2.py"
    ;;
  *)
    echo "usage: $0 [--tmux]" >&2
    exit 1
    ;;
esac
