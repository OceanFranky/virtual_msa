#!/usr/bin/env bash
# Top50（top50_from_top100_prefer_short.csv）× MPNN 采样温度 → 各 1000 条非重复 gap 对齐 MSA
# 输出：test_scaffolds/{ID}/{ID}_mpnn1000_t{T}.a3m（Boltz/评测目录名用 top50 前缀，见下方）
#
#   bash launch_top50_mpnn1000_temp_sweep.sh --tmux
#   bash launch_top50_mpnn1000_temp_sweep.sh --one 8U0X 0.7 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn1000_temp_msa.sh"
CSV="${CSV:-$REPO/top50_from_top100_prefer_short.csv}"

# 用户指定顺序；后续 Boltz/评测：results_boltz_top50_mpnn1000_t* / results_eval_boltz_top50_mpnn1000_t*
TEMPS=(0.7 0.1 1.0 0.01 2.0)
if [[ -n "${GPUS:-}" ]] && [[ "${GPUS}" != *"("* ]]; then
  read -ra GPUS <<< "${GPUS}"
else
  GPUS=(${GPUS[@]:-3 4 5})
fi

read_top50_ids() {
  local py="/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
  [[ -x "$py" ]] || py="python3"
  mapfile -t TOP50_IDS < <("$py" - "$CSV" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        print(row["PDB_ID"].strip())
PY
)
  if [[ ${#TOP50_IDS[@]} -lt 1 ]]; then
    echo "failed to read PDB_ID from $CSV" >&2
    exit 1
  fi
  echo "Loaded ${#TOP50_IDS[@]} PDB from $CSV"
}

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/mpnn1000_top50_gpu${gpu}.lock"
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

run_temp() {
  local temp="$1"
  echo ""
  echo "########## Top50  T=$temp  (${#TOP50_IDS[@]} PDB, GPUs=${GPUS[*]}) ##########"
  local pids=()
  local idx=0
  for id in "${TOP50_IDS[@]}"; do
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    echo ">>> start $id T=$temp gpu=$gpu"
    run_one "$id" "$temp" "$gpu" &
    pids+=($!)
    idx=$((idx + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ "$fail" -eq 0 ]] || { echo "FAIL some jobs at T=$temp" >&2; return 1; }
  echo "########## T=$temp done ##########"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-mpnn1000_temp_top50}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    _gpus="${GPUS[*]:-3 4 5}"
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
GPUS=$(printf %q "$_gpus") bash $(printf %q "$SCRIPT_DIR/launch_top50_mpnn1000_temp_sweep.sh"); \
echo ALL DONE top50 MSA sweep; exec bash"
    tmux new-session -d -s "$SESSION" -n "top50_msa" "$inner"
    echo "tmux: $SESSION"
    echo "  温度: ${TEMPS[*]}"
    echo "  GPU: ${_gpus}"
    echo "  MSA: test_scaffolds/{ID}/{ID}_mpnn1000_t*.a3m"
    echo "  未来 Boltz: results_boltz_top50_mpnn1000_t*"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    read_top50_ids
    run_one "${2:?id}" "${3:?temp}" "${4:-3}"
    ;;
  "")
    read_top50_ids
    for temp in "${TEMPS[@]}"; do
      run_temp "$temp"
    done
    echo "ALL DONE top50 mpnn1000 temp sweep"
    ;;
  *)
    echo "usage: $0 [--tmux | --one ID TEMP GPU]" >&2
    exit 1
    ;;
esac
