#!/usr/bin/env bash
# Top9 × mpnn50 × 温度梯度：各 50 条非重复 → gap 对齐 a3m → 汇总 score → 删除测试 MSA
#
#   bash launch_top9_mpnn50_temp_score_sweep.sh
#   bash launch_top9_mpnn50_temp_score_sweep.sh --one 8BAU 0.5 0
#   bash launch_top9_mpnn50_temp_score_sweep.sh --summarize-only   # 已有 a3m 时只统计
#   bash launch_top9_mpnn50_temp_score_sweep.sh --no-cleanup       # 保留 a3m/npz
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn1000_temp_msa.sh"
SUMMARY_PY="$SCRIPT_DIR/summarize_mpnn_a3m_scores.py"
_PH_PY="/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
PYTHON="${PYTHON:-$_PH_PY}"

TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)
# 与 mpnn1000 温度扫参一致（含 t0.01）
TEMPS=(0.01 0.1 0.2 0.5 0.7 1.0 1.5 2.0)
MSA_N="${MSA_N:-50}"
GPUS=(${GPUS:-0 6 2 4})
OUT_DIR="${OUT_DIR:-$REPO/test_scaffolds/scripts/top9_mpnn50_score_sweep}"
OUT_CSV="$OUT_DIR/top9_mpnn50_score_ranges_by_temp.csv"
OUT_DETAIL="$OUT_DIR/top9_mpnn50_score_ranges_per_pdb.csv"

t_tag() {
  case "$1" in
    2.0|2)   echo "t2.0" ;;
    1.5)     echo "t1.5" ;;
    1.0|1)   echo "t1.0" ;;
    0.7)     echo "t0.7" ;;
    0.5)     echo "t0.5" ;;
    0.2)     echo "t0.2" ;;
    0.1|.1)  echo "t0.1" ;;
    0.01)    echo "t0.01" ;;
    *)       echo "t${1}" ;;
  esac
}

_gpu_lock() {
  local gpu="$1"
  exec 9>"/tmp/mpnn50_top9_gpu${gpu}.lock"
  flock 9
}

run_one() {
  local id="$1"
  local temp="$2"
  local gpu="${3:-0}"
  local tag
  tag=$(t_tag "$temp")
  _gpu_lock "$gpu"
  export CASE_DIR="$ROOT/$id" CASE_ID="$id" TEMPERATURE="$temp"
  export TARGET_SEQS="$MSA_N" MSA_N="$MSA_N"
  export OUT_TAG="_mpnn${MSA_N}_${tag}"
  export WORK="$CASE_DIR/mpnn_work_${MSA_N}_${tag}"
  export RAW_A3M="$CASE_DIR/${CASE_ID}${OUT_TAG}_raw.a3m"
  export BATCH_SIZE="${BATCH_SIZE:-50}" BATCHES_PER_ROUND="${BATCHES_PER_ROUND:-4}" MAX_ROUNDS="${MAX_ROUNDS:-20}"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"
}

cleanup_msas() {
  echo "==> cleanup test MSA (mpnn${MSA_N}_t*)"
  for temp in "${TEMPS[@]}"; do
    local tag
    tag=$(t_tag "$temp")
    for id in "${TOP9[@]}"; do
      local d="$ROOT/$id"
      rm -f "$d/${id}_mpnn${MSA_N}_${tag}.a3m" \
            "$d/${id}_mpnn${MSA_N}_${tag}.npz" \
            "$d/${id}_mpnn${MSA_N}_${tag}_raw.a3m" \
            "$d/${id}_mpnn${MSA_N}_${tag}_raw.npz" \
            "$d/${id}_mpnn${MSA_N}_${tag}.log" \
            "$d/${id}_mpnn${MSA_N}_${tag}_raw.log" 2>/dev/null || true
      rm -rf "$d/mpnn_work_${MSA_N}_${tag}" 2>/dev/null || true
    done
  done
}

summarize() {
  mkdir -p "$OUT_DIR"
  "$PYTHON" "$SUMMARY_PY" \
    --repo "$REPO" \
    --root "$ROOT" \
    --msa-n "$MSA_N" \
    --temps "${TEMPS[@]}" \
    --out-csv "$OUT_CSV" \
    --out-detail-csv "$OUT_DETAIL"
}

run_all_generate() {
  local pids=() idx=0
  for temp in "${TEMPS[@]}"; do
    for id in "${TOP9[@]}"; do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      echo ">>> start $id T=$temp gpu=$gpu"
      run_one "$id" "$temp" "$gpu" &
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ "$fail" -eq 0 ]] || { echo "FAIL: some MPNN jobs failed" >&2; exit 1; }
}

case "${1:-}" in
  --one)
    run_one "${2:?id}" "${3:?temp}" "${4:-0}"
    ;;
  --summarize-only)
    summarize
    ;;
  --cleanup-only)
    cleanup_msas
    ;;
  --no-cleanup)
    run_all_generate
    summarize
  ;;
  "")
    run_all_generate
    summarize
    cleanup_msas
    echo ""
    echo "DONE. 汇总: $OUT_CSV"
    echo "       明细: $OUT_DETAIL"
    ;;
  *)
    echo "usage: $0 [--one ID TEMP GPU | --summarize-only | --cleanup-only | --no-cleanup]" >&2
    exit 1
    ;;
esac
