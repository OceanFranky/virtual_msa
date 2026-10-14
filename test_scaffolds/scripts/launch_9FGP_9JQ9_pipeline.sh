#!/usr/bin/env bash
# 9FGP–9IJC 全流程（5 蛋白 × 5 GPU，一一对应；不跑 9IUH / 9J0V / 9JQ9）
# MPNN 3000 → boltz_ph 10 cycles → test_scaffords_5.20
#
#   bash launch_9FGP_9JQ9_pipeline.sh --tmux-mpnn
#   bash launch_9FGP_9JQ9_pipeline.sh --tmux-boltz
#   bash launch_9FGP_9JQ9_pipeline.sh --status
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 仅 5 个：GPU0–4 各跑一个（跳过 9IUH 9J0V 9JQ9）
export IDS="${IDS:-9FGP 9FOW 9FYM 9GSO 9IJC}"
export NGPU="${NGPU:-5}"

MPNN="$SCRIPT_DIR/launch_scaffold_series_mpnn3000.sh"
BOLTZ="$SCRIPT_DIR/launch_scaffold_series_boltz.sh"

case "${1:-}" in
  --tmux-mpnn)
    export SESSION="${SESSION:-mpnn_9FGP_9JQ9}"
    FORCE="${FORCE:-1}" bash "$MPNN" --tmux
    ;;
  --tmux-boltz)
    export SESSION="${SESSION:-boltz_9FGP_9JQ9}"
    FORCE="${FORCE:-1}" bash "$BOLTZ" --tmux
    ;;
  --status)
    for id in $IDS; do
      npz="/data/wenhao/Protein-Hunter-server/test_scaffolds/$id/${id}_mpnn3000_unique.npz"
      best="/data/wenhao/Protein-Hunter-server/test_scaffords_5.20/$id/0_protein_hunter_design/run_0/${id}_run_0_best_structure.pdb"
      printf "%-6s mpnn3000=%s  boltz_best=%s\n" "$id" \
        "$([ -f "$npz" ] && echo OK || echo --)" \
        "$([ -f "$best" ] && echo OK || echo --)"
    done
    ;;
  *)
    echo "usage: $0 [--tmux-mpnn|--tmux-boltz|--status]" >&2
    exit 1
    ;;
esac
