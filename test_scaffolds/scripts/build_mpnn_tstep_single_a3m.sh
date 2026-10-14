#!/usr/bin/env bash
# 单 scaffold：T=0.1,0.2,...,2.0（步长 0.1）各 SEQS_PER_TEMP 条非重复 → 合并为一个 gap 对齐 a3m/npz
#
#   CASE_DIR=.../8BAU bash build_mpnn_tstep_single_a3m.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
BUILD_ONE="$SCRIPT_DIR/build_mpnn_msa_unique.sh"
MERGE_PY="$SCRIPT_DIR/merge_mpnn_a3m_records.py"
ALIGN_PY="$SCRIPT_DIR/align_mpnn_a3m_to_native.py"

CASE_DIR="${CASE_DIR:?set CASE_DIR}"
CASE_ID="${CASE_ID:-$(basename "$CASE_DIR")}"
SEQS_PER_TEMP="${SEQS_PER_TEMP:-50}"
TEMP_START="${TEMP_START:-0.1}"
TEMP_END="${TEMP_END:-2.0}"
TEMP_STEP="${TEMP_STEP:-0.1}"
OUT_TAG="${OUT_TAG:-_mpnn${SEQS_PER_TEMP}_t01_20}"
FINAL_OUT_TAG="$OUT_TAG"
WORK="${WORK:-$CASE_DIR/mpnn_work_t01_20}"
BASE_WORK="$WORK"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-1}"
MERGE_ONLY="${MERGE_ONLY:-0}"

_PH_PY="/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
PYTHON="${PYTHON:-$_PH_PY}"

PDB="$CASE_DIR/${CASE_ID}.pdb"
NATIVE_A3M="$CASE_DIR/${CASE_ID}.a3m"
for f in "$PDB" "$NATIVE_A3M"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

mkdir -p "$WORK/per_temp"
PER_TEMP_A3MS=()

mapfile -t TEMPS < <(
  "$PYTHON" - "$TEMP_START" "$TEMP_END" "$TEMP_STEP" <<'PY'
import sys
start, end, step = map(float, sys.argv[1:4])
t = start
while t <= end + 1e-9:
    print(f"{t:.1f}")
    t = round(t + step, 1)
PY
)

echo "=== build_mpnn_tstep ${CASE_ID} temps=${#TEMPS[@]} x ${SEQS_PER_TEMP} -> ${OUT_TAG} $(date -Is) ==="

for temp in "${TEMPS[@]}"; do
  tkey="${temp//./}"
  tdir="$BASE_WORK/per_temp/t${tkey}"
  tout_tag="_mpnn${SEQS_PER_TEMP}_t${temp}_part"
  tout_a3m="$CASE_DIR/${CASE_ID}${tout_tag}.a3m"

  if [[ -f "$tout_a3m" ]] && [[ "$(grep -c '^>' "$tout_a3m" || true)" -ge $((SEQS_PER_TEMP + 1)) ]]; then
    echo "skip T=$temp (exists $tout_a3m)"
    PER_TEMP_A3MS+=("$tout_a3m")
    continue
  fi
  if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "missing part for T=$temp: $tout_a3m" >&2
    exit 1
  fi

  echo ">>> T=$temp  target=$SEQS_PER_TEMP"
  export CASE_DIR CASE_ID TEMPERATURE="$temp" TARGET_SEQS="$SEQS_PER_TEMP"
  export OUT_TAG="$tout_tag" WORK="$tdir"
  export BATCH_SIZE="${BATCH_SIZE:-50}" BATCHES_PER_ROUND="${BATCHES_PER_ROUND:-4}" MAX_ROUNDS="${MAX_ROUNDS:-20}"
  export SKIP_MPNN="${SKIP_MPNN:-0}"
  bash "$BUILD_ONE"
  OUT_TAG="$FINAL_OUT_TAG"

  PER_TEMP_A3MS+=("$tout_a3m")
done

RAW_A3M="$CASE_DIR/${CASE_ID}${FINAL_OUT_TAG}_raw.a3m"
OUT_A3M="$CASE_DIR/${CASE_ID}${FINAL_OUT_TAG}.a3m"
OUT_NPZ="$CASE_DIR/${CASE_ID}${FINAL_OUT_TAG}.npz"

MERGE_ARGS=(--out-a3m "$RAW_A3M")
for p in "${PER_TEMP_A3MS[@]}"; do
  MERGE_ARGS+=(--src-a3m "$p")
done
"$PYTHON" "$MERGE_PY" "${MERGE_ARGS[@]}"

TARGET_TOTAL=$(grep -c '^>' "$RAW_A3M" || true)
TARGET_TOTAL=$((TARGET_TOTAL - 1))
echo "==> gap align n=$TARGET_TOTAL to native"
"$PYTHON" "$ALIGN_PY" \
  --case-dir "$CASE_DIR" \
  --case-id "$CASE_ID" \
  --src-a3m "$RAW_A3M" \
  --out-a3m "$OUT_A3M" \
  --target "$TARGET_TOTAL" \
  --repo "$REPO" \
  --python "$PYTHON"

if [[ "$KEEP_INTERMEDIATE" != "1" ]]; then
  rm -f "$RAW_A3M" "${RAW_A3M%.a3m}.npz" 2>/dev/null || true
  rm -rf "$WORK/per_temp" 2>/dev/null || true
fi

N=$(grep -c '^>' "$OUT_A3M" || true)
echo "DONE ${CASE_ID} ${FINAL_OUT_TAG} headers=$N a3m=$OUT_A3M npz=$OUT_NPZ"
