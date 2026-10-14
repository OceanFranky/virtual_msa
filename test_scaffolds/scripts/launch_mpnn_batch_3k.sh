#!/usr/bin/env bash
# 11/ + test_scaffolds 批量 MPNN 3000（8 GPU 队列）
#
#   bash launch_mpnn_batch_3k.sh              # 生成 pending 列表
#   bash launch_mpnn_batch_3k.sh --tmux       # 8 窗 GPU0-7
#   bash launch_mpnn_batch_3k.sh --status
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="$SCRIPT_DIR/run_mpnn_queue_worker.sh"
QUEUE="$SCRIPT_DIR/mpnn_batch_pending.tsv"
SESSION="${SESSION:-mpnn_batch_3k_x8}"
WORKERS="${WORKERS:-8}"

gen_queue() {
  "$PYTHON_GEN"
}

PYTHON_GEN="/data/wenhao/miniconda3/envs/proteinhunter/bin/python3"
gen_queue() {
  "$PYTHON_GEN" <<'PY'
from pathlib import Path
r11 = Path("/data/wenhao/Protein-Hunter-server/11")
rts = Path("/data/wenhao/Protein-Hunter-server/test_scaffolds")
out = Path("/data/wenhao/Protein-Hunter-server/test_scaffolds/scripts/mpnn_batch_pending.tsv")

def ready(root, cid):
    d = root / cid
    return (d / f"{cid}.pdb").exists() and (d / f"{cid}_mask.pdb").exists()

def has_3k(root, cid):
    return (root / cid / f"{cid}_mpnn3000_unique.npz").exists()

rows = []
for d in sorted(rts.iterdir()):
    if not d.is_dir(): continue
    cid = d.name
    if not ready(rts, cid) or has_3k(rts, cid): continue
    rows.append((cid, str(rts)))
for d in sorted(r11.iterdir()):
    if not d.is_dir(): continue
    cid = d.name
    if not ready(r11, cid) or has_3k(r11, cid) or has_3k(rts, cid): continue
    if any(r[0]==cid for r in rows): continue
    rows.append((cid, str(r11)))
out.write_text("id\troot\n" + "\n".join(f"{a}\t{b}" for a,b in rows) + "\n")
print(f"pending {len(rows)} -> {out}")
PY
}

status() {
  total=$(tail -n +2 "$QUEUE" 2>/dev/null | wc -l)
  done=0
  while IFS=$'\t' read -r cid root; do
    [[ "$cid" == "id" || -z "$cid" ]] && continue
    [[ -f "$root/$cid/${cid}_mpnn3000_unique.npz" ]] && done=$((done+1))
  done < "$QUEUE"
  echo "queue: $QUEUE  total=$total  mpnn3000_done=$done  remain=$((total-done))"
}

MODE="${1:-}"
case "$MODE" in
  --status)
    [[ -f "$QUEUE" ]] || gen_queue
    status
    ;;
  --tmux)
    gen_queue
    status
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    for g in $(seq 0 $((WORKERS - 1))); do
      cmd="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
export PATH=/data/wenhao/miniconda3/envs/proteinhunter/bin:\$PATH; \
GPU=$g WORKERS=$WORKERS bash $(printf %q "$WORKER"); \
echo DONE worker GPU$g; exec bash"
      if [[ "$g" -eq 0 ]]; then
        tmux new-session -d -s "$SESSION" -n "gpu0" "$cmd"
      else
        tmux new-window -t "$SESSION" -n "gpu$g" "$cmd"
      fi
    done
    echo "tmux: $SESSION  ($WORKERS workers, 每卡 1 队列进程)"
    echo "  queue: $QUEUE"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --gen-queue)
    gen_queue
    ;;
  *)
    echo "usage: $0 [--gen-queue|--tmux|--status]" >&2
    exit 1
    ;;
esac
