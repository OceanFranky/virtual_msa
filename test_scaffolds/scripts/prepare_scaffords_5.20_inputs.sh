#!/usr/bin/env bash
# 为 boltz_ph 批处理准备 test_scaffords_5.20 输入（链接 mpnn3000 MSA）
set -euo pipefail

SRC="${SRC:-/data/wenhao/Protein-Hunter-server/test_scaffolds}"
DST="${DST:-/data/wenhao/Protein-Hunter-server/test_scaffords_5.20}"
TAG="${TAG:-_mpnn3000_unique}"

IDS=(8G52 8G53 8G64 8GAC 8GBE)

for id in "${IDS[@]}"; do
  d="$DST/$id"
  mkdir -p "$d"
  for f in pdb fasta; do
    ln -sfn "$SRC/$id/${id}.$f" "$d/${id}.$f"
  done
  ln -sfn "$SRC/$id/${id}_mask.pdb" "$d/${id}_mask.pdb"
  ln -sfn "$SRC/$id/${id}${TAG}.a3m" "$d/${id}.a3m"
  ln -sfn "$SRC/$id/${id}${TAG}.npz" "$d/${id}.npz"
  echo "prepared $d"
done

echo "DONE inputs under $DST"
