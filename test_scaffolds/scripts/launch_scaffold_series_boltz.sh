#!/usr/bin/env bash
# 批量 boltz_ph：3000 MSA、10 cycles、1 run → test_scaffords_5.20
#
#   IDS="9FGP 9FOW ..." bash launch_scaffold_series_boltz.sh --tmux
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
INPUT_ROOT="${INPUT_ROOT:-/data/wenhao/Protein-Hunter-server/test_scaffords_5.20}"
RESULTS_ROOT="${RESULTS_ROOT:-$INPUT_ROOT}"
PREP="$SCRIPT_DIR/prepare_scaffords_inputs.sh"
LAUNCH_ONE="$SCRIPT_DIR/launch_boltz_ph_8G_series.sh"
SESSION="${SESSION:-boltz_scaffold_520}"

DEFAULT_IDS=(9FGP 9FOW 9FYM 9GSO 9IJC)
if [[ -n "${IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ID_ARR=($IDS)
else
  ID_ARR=("${DEFAULT_IDS[@]}")
fi
NGPU="${NGPU:-5}"

export IDS="${ID_ARR[*]}"

MODE="${1:-}"
case "$MODE" in
  --tmux)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    bash "$PREP"
    n=${#ID_ARR[@]}
    if [[ "$n" -gt "$NGPU" ]]; then
      echo "错误: ${n} 个蛋白 > ${NGPU} 块 GPU。请减少 IDS。" >&2
      exit 1
    fi
    i=0
    for id in "${ID_ARR[@]}"; do
      gpu=$i
      cmd="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
export PATH=/data/wenhao/miniconda3/envs/proteinhunter/bin:\$PATH; \
export CLEAN_PARTIAL=1; \
bash $(printf %q "$LAUNCH_ONE") --one $(printf %q "$id") $gpu; \
echo DONE BOLTZ $id; exec bash"
      if [[ "$i" -eq 0 ]]; then
        tmux new-session -d -s "$SESSION" -n "$id" "$cmd"
      else
        tmux new-window -t "$SESSION" -n "$id" "$cmd"
      fi
      i=$((i + 1))
    done
    echo "tmux: $SESSION  (${#ID_ARR[@]} windows = GPU 0–$(( ${#ID_ARR[@]} - 1 )) 一一对应)"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    bash "$PREP"
    bash "$LAUNCH_ONE" --one "${2:?id}" "${3:-0}"
    ;;
  "")
    bash "$PREP"
    i=0
    for id in "${ID_ARR[@]}"; do
      bash "$LAUNCH_ONE" --one "$id" "$i"
      i=$((i + 1))
    done
    ;;
  *)
    echo "usage: IDS='...' $0 [--tmux|--one ID GPU]" >&2
    exit 1
    ;;
esac
