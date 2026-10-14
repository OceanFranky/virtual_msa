#!/usr/bin/env bash
# Top10 难合成蛋白：各 3000 条不重复 MPNN MSA → test_scaffolds/{ID}/
# 后续 boltz 输出到 test_scaffords_5.20
#
#   bash launch_top10_hard_mpnn3000.sh --tmux
#   FORCE=1 bash launch_top10_hard_mpnn3000.sh --tmux
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-/data/wenhao/Protein-Hunter-server/test_scaffolds}"
NGPU="${NGPU:-16}"
SESSION="${SESSION:-mpnn_top10_hard_3k}"

# Rank 1–10（难合成 benchmark）
export IDS="${IDS:-8U0X 8IQC 8ADD 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX}"

# 8EO5 仅在 11/ 目录，补链到 test_scaffolds
prep_8eo5() {
  local id=8EO5 src="/data/wenhao/Protein-Hunter-server/11/8EO5"
  local dst="$ROOT/8EO5"
  if [[ ! -f "$dst/8EO5.pdb" ]] && [[ -f "$src/8EO5.pdb" ]]; then
    mkdir -p "$dst"
    for f in pdb fasta; do ln -sfn "$src/8EO5.$f" "$dst/8EO5.$f"; done
    ln -sfn "$src/8EO5_mask.pdb" "$dst/8EO5_mask.pdb"
    echo "linked 8EO5 from $src -> $dst"
  fi
}

prep_8eo5

case "${1:-}" in
  --status)
    for id in $IDS; do
      npz="$ROOT/$id/${id}_mpnn3000_unique.npz"
      n=$(grep -c '^>' "$ROOT/$id/${id}_mpnn3000_unique.a3m" 2>/dev/null || echo 0)
      printf "%-6s mpnn3000=%s  a3m_headers=%s\n" "$id" \
        "$([ -f "$npz" ] && echo OK || echo --)" "$((n>0?n-1:0))"
    done
    ;;
  --tmux|--one)
    exec env NGPU="$NGPU" SESSION="$SESSION" bash "$SCRIPT_DIR/launch_scaffold_series_mpnn3000.sh" "$@"
    ;;
  *)
    echo "usage: $0 [--tmux|--one ID GPU|--status]" >&2
    exit 1
    ;;
esac
