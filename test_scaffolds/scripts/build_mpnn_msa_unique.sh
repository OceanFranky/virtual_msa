#!/usr/bin/env bash
# Monomer scaffold: ProteinMPNN → 3000 unique sequences → a3m → npz
#
#   CASE_DIR=/path/to/test_scaffolds/8G52  CASE_ID=8G52  bash build_mpnn_msa_unique.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
MPNN_ROOT="${MPNN_ROOT:-$REPO/LigandMPNN}"
CKPT="${CKPT:-$MPNN_ROOT/model_params/proteinmpnn_v_48_020.pt}"
CONVERTER="${CONVERTER:-$REPO/protein_sequence_design_tool/convert_a3m_to_npz_local.py}"

CASE_DIR="${CASE_DIR:?set CASE_DIR}"
CASE_ID="${CASE_ID:-$(basename "$CASE_DIR")}"
PDB="${PDB:-$CASE_DIR/${CASE_ID}.pdb}"
FASTA="${FASTA:-$CASE_DIR/${CASE_ID}.fasta}"
DESIGN_CHAIN="${DESIGN_CHAIN:-A}"
TARGET_SEQS="${TARGET_SEQS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-100}"
BATCHES_PER_ROUND="${BATCHES_PER_ROUND:-50}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
OUT_TAG="${OUT_TAG:-_mpnn${TARGET_SEQS}_unique}"
WORK="${WORK:-$CASE_DIR/mpnn_work_${TARGET_SEQS}_unique}"
SKIP_MPNN="${SKIP_MPNN:-0}"

_PH_PY="/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$_PH_PY" ]] && "$_PH_PY" -c "import torch" 2>/dev/null; then
    PYTHON="$_PH_PY"
  else
    PYTHON="python3"
  fi
fi

for f in "$PDB" "$FASTA" "$CKPT"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

# MPNN 按 PDB 设计；长度以 PDB CA 为准（与 fasta 可能差几个残基）
PROTEIN_LEN=$(grep -cE '^ATOM.{16}CA ' "$PDB" || true)
if [[ "$PROTEIN_LEN" -lt 1 ]]; then
  PROTEIN_LEN=$(grep -cE '^ATOM.* CA ' "$PDB" || true)
fi
if [[ "$PROTEIN_LEN" -lt 1 ]]; then
  echo "cannot count CA in $PDB" >&2
  exit 1
fi

OUT_A3M="$CASE_DIR/${CASE_ID}${OUT_TAG}.a3m"
OUT_NPZ="$CASE_DIR/${CASE_ID}${OUT_TAG}.npz"
STAMP="${CASE_ID}${OUT_TAG}"
LOG="$CASE_DIR/${CASE_ID}${OUT_TAG}.log"

exec > >(tee -a "$LOG") 2>&1

echo "=== $CASE_ID  target_unique=$TARGET_SEQS  len=$PROTEIN_LEN  $(date -Is) ==="
echo "PDB=$PDB  chain=$DESIGN_CHAIN  WORK=$WORK"

mkdir -p "$WORK/rounds"

if [[ "$SKIP_MPNN" != "1" ]]; then
  seed=0
  for round in $(seq 1 "$MAX_ROUNDS"); do
    # count unique so far
    mapfile -t EXISTING_FAS < <(find "$WORK/rounds" -name '*.fa' 2>/dev/null | sort)
    if [[ "${#EXISTING_FAS[@]}" -gt 0 ]]; then
      n_uniq=$("$PYTHON" - "$SCRIPT_DIR" "$WORK/rounds" "$TARGET_SEQS" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from mpnn_fa_to_a3m_monomer import collect_unique_from_fas
root = Path(sys.argv[2])
target = int(sys.argv[3])
fas = sorted(root.glob("**/*.fa"))
des, _ = collect_unique_from_fas(fas, target)
print(len(des))
PY
)
    else
      n_uniq=0
    fi
    echo "round $round  unique_so_far=$n_uniq / $TARGET_SEQS"
    if [[ "$n_uniq" -ge "$TARGET_SEQS" ]]; then
      break
    fi

    rdir="$WORK/rounds/round_${round}_seed_${seed}"
    mkdir -p "$rdir"
    echo "==> MPNN round=$round seed=$seed batches=$BATCHES_PER_ROUND x $BATCH_SIZE"
    t0=$(date +%s)
    "$PYTHON" "$MPNN_ROOT/run.py" \
      --model_type protein_mpnn \
      --checkpoint_protein_mpnn "$CKPT" \
      --pdb_path "$PDB" \
      --out_folder "$rdir" \
      --chains_to_design "$DESIGN_CHAIN" \
      --batch_size "$BATCH_SIZE" \
      --number_of_batches "$BATCHES_PER_ROUND" \
      --temperature "$TEMPERATURE" \
      --seed "$seed"
    echo "MPNN round $round done in $(( $(date +%s) - t0 ))s"
    seed=$((seed + 1))
  done
fi

mapfile -t ALL_FAS < <(find "$WORK/rounds" -name '*.fa' | sort)
if [[ "${#ALL_FAS[@]}" -eq 0 ]]; then
  echo "no fa under $WORK/rounds" >&2
  exit 1
fi

echo "==> fa → a3m (dedupe, require $TARGET_SEQS unique)"
FA_ARGS=()
for f in "${ALL_FAS[@]}"; do
  FA_ARGS+=(--fa "$f")
done
FIRST_FA="${ALL_FAS[0]}"
"$PYTHON" "$SCRIPT_DIR/mpnn_fa_to_a3m_monomer.py" \
  "${FA_ARGS[@]}" \
  --fasta "$FASTA" \
  --query-from-fa "$FIRST_FA" \
  --out-a3m "$OUT_A3M" \
  --query-name "$CASE_ID" \
  --target-seqs "$TARGET_SEQS"

echo "==> a3m → npz"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT
mkdir -p "$TMP_ROOT/$STAMP"
cp "$OUT_A3M" "$TMP_ROOT/$STAMP/${STAMP}.a3m"
export PYTHONPATH="${REPO}/boltz_ph/src:${PYTHONPATH:-}"
"$PYTHON" "$CONVERTER" --scaffold-root "$TMP_ROOT"
cp "$TMP_ROOT/$STAMP/${STAMP}.npz" "$OUT_NPZ"

N=$(grep -c '^>' "$OUT_A3M" || true)
UNIQ=$((N - 1))
echo "DONE $(date -Is)"
echo "  a3m: $OUT_A3M  (headers=$N  designed=$UNIQ)"
echo "  npz: $OUT_NPZ"
echo "  log: $LOG"
