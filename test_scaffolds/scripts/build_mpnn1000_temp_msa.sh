#!/usr/bin/env bash
# Top9 单体：ProteinMPNN @ 指定温度 → 1000 条非重复 → gap 对齐到原生 a3m query → a3m/npz
#
#   TEMPERATURE=1.0 CASE_DIR=.../8BAU bash build_mpnn1000_temp_msa.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"

CASE_DIR="${CASE_DIR:?set CASE_DIR}"
CASE_ID="${CASE_ID:-$(basename "$CASE_DIR")}"
TEMPERATURE="${TEMPERATURE:?set TEMPERATURE}"
TARGET_SEQS="${TARGET_SEQS:-1000}"

# 输出后缀：_mpnn1000_t1.0 / _mpnn1000_t0.01
case "$TEMPERATURE" in
  2.0|2)   T_TAG="t2.0" ;;
  1.5)     T_TAG="t1.5" ;;
  1.0|1)   T_TAG="t1.0" ;;
  0.7)     T_TAG="t0.7" ;;
  0.5)     T_TAG="t0.5" ;;
  0.2)     T_TAG="t0.2" ;;
  0.1|.1)   T_TAG="t0.1" ;;
  0.01)    T_TAG="t0.01" ;;
  *)       T_TAG="t${TEMPERATURE}" ;;
esac

OUT_TAG="${OUT_TAG:-_mpnn1000_${T_TAG}}"
RAW_TAG="${OUT_TAG}_raw"
WORK="${WORK:-$CASE_DIR/mpnn_work_1000_${T_TAG}}"
RAW_A3M="${RAW_A3M:-$CASE_DIR/${CASE_ID}${RAW_TAG}.a3m}"

export CASE_DIR CASE_ID TEMPERATURE TARGET_SEQS OUT_TAG WORK RAW_A3M
export TARGET_SEQS BATCH_SIZE="${BATCH_SIZE:-100}" BATCHES_PER_ROUND="${BATCHES_PER_ROUND:-80}"
export MAX_ROUNDS="${MAX_ROUNDS:-50}"
export SKIP_MPNN="${SKIP_MPNN:-0}"

PDB="$CASE_DIR/${CASE_ID}.pdb"
NATIVE_A3M="$CASE_DIR/${CASE_ID}.a3m"
for f in "$PDB" "$NATIVE_A3M"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

_PH_PY="/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
PYTHON="${PYTHON:-$_PH_PY}"

echo "=== build_mpnn1000 ${CASE_ID} T=${TEMPERATURE} tag=${OUT_TAG} $(date -Is) ==="

# Step 1: MPNN + raw a3m（query 用 PDB 链长，供 observed 推断）
SKIP_MPNN="$SKIP_MPNN" \
  CASE_DIR="$CASE_DIR" CASE_ID="$CASE_ID" \
  TARGET_SEQS="$TARGET_SEQS" TEMPERATURE="$TEMPERATURE" \
  OUT_TAG="$RAW_TAG" WORK="$WORK" \
  bash "$SCRIPT_DIR/build_mpnn_msa_unique.sh"

BUILT_RAW="$RAW_A3M"
[[ -f "$BUILT_RAW" ]] || { echo "missing raw a3m: $BUILT_RAW" >&2; exit 1; }
OUT_A3M="$CASE_DIR/${CASE_ID}${OUT_TAG}.a3m"
OUT_NPZ="$CASE_DIR/${CASE_ID}${OUT_TAG}.npz"

echo "==> gap align to native ${NATIVE_A3M}"
"$PYTHON" "$SCRIPT_DIR/align_mpnn_a3m_to_native.py" \
  --case-dir "$CASE_DIR" \
  --case-id "$CASE_ID" \
  --src-a3m "$BUILT_RAW" \
  --out-a3m "$OUT_A3M" \
  --target "$TARGET_SEQS" \
  --repo "$REPO" \
  --python "$PYTHON"

# npz 已在 align 脚本生成；删除 raw 可选
if [[ "${KEEP_RAW:-0}" != "1" ]]; then
  rm -f "$BUILT_RAW" "${BUILT_RAW%.a3m}.npz" 2>/dev/null || true
fi

N=$(grep -c '^>' "$OUT_A3M" || true)
echo "DONE ${CASE_ID} ${OUT_TAG} headers=$N a3m=$OUT_A3M npz=$OUT_NPZ"
