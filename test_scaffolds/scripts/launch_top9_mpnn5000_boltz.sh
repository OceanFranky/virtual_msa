#!/usr/bin/env bash
# Top9 + mpnn5000_unique → results_boltz_top9_mpnn5000（多卡，每卡 2 进程）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FORCE="${FORCE:-1}" bash "$SCRIPT_DIR/launch_top9_mpnn5000_parallel.sh" --stop-old
FORCE="${FORCE:-1}" bash "$SCRIPT_DIR/launch_top9_mpnn5000_parallel.sh" --tmux
