#!/usr/bin/env bash
# 为 boltz_ph 准备 test_scaffords_5.20 输入（链接 mpnn3000 MSA）
#   IDS="9FGP 9FOW" bash prepare_scaffords_inputs.sh
set -euo pipefail

SRC="${SRC:-/data/wenhao/Protein-Hunter-server/test_scaffolds}"
DST="${DST:-/data/wenhao/Protein-Hunter-server/test_scaffords_5.20}"
TAG="${TAG:-_mpnn3000_unique}"

if [[ -n "${IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ID_ARR=($IDS)
else
  ID_ARR=(8G52 8G53 8G64 8GAC 8GBE)
fi

for id in "${ID_ARR[@]}"; do
  d="$DST/$id"
  mkdir -p "$d"
  for f in pdb fasta; do
    ln -sfn "$SRC/$id/${id}.$f" "$d/${id}.$f"
  done
  ln -sfn "$SRC/$id/${id}_mask.pdb" "$d/${id}_mask.pdb"
  if [[ ! -f "$SRC/$id/${id}${TAG}.a3m" ]] || [[ ! -f "$SRC/$id/${id}${TAG}.npz" ]]; then
    echo "missing MPNN output: $SRC/$id/${id}${TAG}.{a3m,npz}" >&2
    exit 1
  fi
  ln -sfn "$SRC/$id/${id}${TAG}.a3m" "$d/${id}.a3m"
  ln -sfn "$SRC/$id/${id}${TAG}.npz" "$d/${id}.npz"
  echo "prepared $d"
done

echo "DONE inputs under $DST (${#ID_ARR[@]} cases)"
