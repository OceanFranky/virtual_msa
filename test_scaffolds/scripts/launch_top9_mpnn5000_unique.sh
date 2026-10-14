#!/usr/bin/env bash
# Top9：ProteinMPNN → 5000 条非重复序列 → {ID}_mpnn5000_unique.a3m / .npz
#
#   bash launch_top9_mpnn5000_unique.sh --tmux     # 9 窗，GPU 0–8 各 1 个
#   bash launch_top9_mpnn5000_unique.sh 8U0X 0   # 单个
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn_msa_unique.sh"

# 经典 Top9（不含 8ADD）
IDS=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
# 优先用空闲卡 2–5；同卡多任务用 flock 排队
GPUS=(2 3 4 5 2 3 4 5 2)

export TARGET_SEQS=5000
export OUT_TAG=_mpnn5000_unique
export BATCH_SIZE=100
export BATCHES_PER_ROUND=80
export MAX_ROUNDS=40
export TEMPERATURE=0.1

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/mpnn5000_gpu${gpu}.lock"
  exec 9>"$lock"
  echo "  waiting GPU $gpu lock ($lock)..."
  flock 9
}

run_one() {
  local id="$1"
  local gpu="${2:-0}"
  _gpu_lock "$gpu"
  export CASE_DIR="$ROOT/$id"
  export CASE_ID="$id"
  export PDB="$CASE_DIR/${id}.pdb"
  export FASTA="$CASE_DIR/${id}.fasta"
  export WORK="$CASE_DIR/mpnn_work_5000_unique"
  echo "===== $id  gpu=$gpu  TARGET=$TARGET_SEQS  $(date -Is) ====="
  CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"
}

_tmux_one() {
  local id="$1"
  local gpu="$2"
  local SESS="mpnn5000_${id}"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "skip existing tmux: $SESS" >&2
    return 1
  fi
  local cmd
  cmd=$(printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; bash %q %q %s 2>&1 | tee -a %q; echo DONE %s; exec bash' \
    "$SCRIPT_DIR/launch_top9_mpnn5000_unique.sh" "$id" "$gpu" \
    "$ROOT/$id/${id}_mpnn5000_unique.log" "$id")
  tmux new-session -d -s "$SESS" -n "$id" "$cmd"
  echo "tmux: $SESS  GPU=$gpu  log=$ROOT/$id/${id}_mpnn5000_unique.log"
}

case "${1:-}" in
  --tmux)
    i=0
    for id in "${IDS[@]}"; do
      gpu="${GPUS[$i]:-0}"
      _tmux_one "$id" "$gpu" || true
      i=$((i + 1))
    done
    echo "attach: tmux ls | grep mpnn5000_"
    ;;
  "")
    echo "usage: $0 --tmux | <PDB_ID> <gpu>" >&2
    exit 1
    ;;
  *)
    run_one "$1" "${2:-0}"
    ;;
esac
