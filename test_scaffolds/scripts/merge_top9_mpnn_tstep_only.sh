#!/usr/bin/env bash
# 仅合并已有 per-temp part a3m → 单个 _mpnn50_t01_20.a3m（MPNN 已跑完时用）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn_tstep_single_a3m.sh"
TOP9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)

for id in "${TOP9[@]}"; do
  echo ">>> merge $id"
  MERGE_ONLY=1 SKIP_MPNN=1 CASE_DIR="$ROOT/$id" CASE_ID="$id" bash "$BUILD"
done

echo "ALL MERGED"
for id in "${TOP9[@]}"; do
  f="$ROOT/$id/${id}_mpnn50_t01_20.a3m"
  n=$(grep -c '^>' "$f" 2>/dev/null || echo 0)
  echo "  $id  headers=$n  $f"
done
