#!/usr/bin/env bash
# Top50 未完成 MSA 补跑：GPU 6/7，每卡 2 进程；跳过已完成 (1001 headers)
#
#   bash launch_top50_mpnn1000_resume_gpu67.sh --tmux
#   bash launch_top50_mpnn1000_resume_gpu67.sh --one 8U0X 0.1 6
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
BUILD="$SCRIPT_DIR/build_mpnn1000_temp_msa.sh"
CSV="${CSV:-$REPO/top50_from_top100_prefer_short.csv}"
TEMPS=(0.7 0.1 1.0 0.01 2.0)

if [[ -n "${GPUS:-}" ]] && [[ "${GPUS}" != *"("* ]]; then
  read -ra GPUS <<< "${GPUS}"
else
  GPUS=(${GPUS[@]:-6 7})
fi
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"

is_done() {
  local id="$1" ttag="$2"
  local a3m="$ROOT/$id/${id}_mpnn1000_${ttag}.a3m"
  local npz="$ROOT/$id/${id}_mpnn1000_${ttag}.npz"
  [[ -f "$a3m" && -f "$npz" ]] || return 1
  local n
  n=$(grep -c '^>' "$a3m" 2>/dev/null || echo 0)
  [[ "$n" -eq 1001 ]]
}

temp_tag() {
  case "$1" in
    2|2.0) echo t2.0 ;;
    1.5) echo t1.5 ;;
    1|1.0) echo t1.0 ;;
    0.7) echo t0.7 ;;
    0.5) echo t0.5 ;;
    0.2) echo t0.2 ;;
    0.1|.1) echo t0.1 ;;
    0.01) echo t0.01 ;;
    *) echo "t$1" ;;
  esac
}

build_pending() {
  PENDING=()
  mapfile -t TOP50_IDS < <("$PY" - "$CSV" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        print(row["PDB_ID"].strip())
PY
)
  for temp in "${TEMPS[@]}"; do
    local ttag
    ttag=$(temp_tag "$temp")
    for id in "${TOP50_IDS[@]}"; do
      if is_done "$id" "$ttag"; then
        continue
      fi
      PENDING+=("${id}:${temp}")
    done
  done
  echo "Pending MSA jobs: ${#PENDING[@]}  (GPUs=${GPUS[*]} procs/gpu=$PROCS_PER_GPU)"
}

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/mpnn1000_top50_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one_job() {
  local id="$1"
  local temp="$2"
  local gpu="$3"
  _gpu_lock "$gpu"

  export CASE_DIR="$ROOT/$id" CASE_ID="$id" TEMPERATURE="$temp"
  export DESIGN_CHAIN="${DESIGN_CHAIN:-A}"
  export PDB="$ROOT/$id/${id}.pdb"

  # 特殊 case
  if [[ "$id" == "8QOH" ]]; then
    export DESIGN_CHAIN=B
  fi
  if [[ "$id" == "8B2S" && -f "$ROOT/$id/${id}_mask.pdb" ]]; then
    export PDB="$ROOT/$id/${id}_mask.pdb"
    export DESIGN_CHAIN=A
  fi

  echo "=== RUN $id T=$temp gpu=$gpu chain=$DESIGN_CHAIN pdb=$PDB ==="
  if CUDA_VISIBLE_DEVICES="$gpu" bash "$BUILD"; then
    echo "OK $id T=$temp"
  else
    echo "FAIL $id T=$temp (see logs under $ROOT/$id/)" >&2
    return 1
  fi
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local jobs=("$@")
  local batch=()
  for job in "${jobs[@]}"; do
    batch+=("$job")
    if [[ ${#batch[@]} -ge $PROCS_PER_GPU ]]; then
      local pids=()
      for j in "${batch[@]}"; do
        local id="${j%%:*}" temp="${j#*:}"
        echo ">>> gpu=$gpu  $id T=$temp"
        run_one_job "$id" "$temp" "$gpu" &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    local pids=()
    for j in "${batch[@]}"; do
      local id="${j%%:*}" temp="${j#*:}"
      echo ">>> gpu=$gpu  $id T=$temp"
      run_one_job "$id" "$temp" "$gpu" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
}

assign_and_run() {
  build_pending
  [[ ${#PENDING[@]} -gt 0 ]] || { echo "Nothing pending."; return 0; }

  declare -A QUEUE
  for g in "${GPUS[@]}"; do QUEUE[$g]=""; done
  local i=0
  for job in "${PENDING[@]}"; do
    local g="${GPUS[$((i % ${#GPUS[@]}))]}"
    QUEUE[$g]+="$job "
    i=$((i + 1))
  done

  for g in "${GPUS[@]}"; do
    # shellcheck disable=SC2206
    local jobs=(${QUEUE[$g]})
    [[ ${#jobs[@]} -eq 0 ]] && continue
    echo "######## GPU $g  ${#jobs[@]} jobs ########"
    run_gpu_queue "$g" "${jobs[@]}"
  done
}

stop_boltz_on_gpus() {
  if [[ "${STOP_BOLTZ_GPU67:-0}" != "1" ]]; then
    echo "Not stopping Boltz on GPU 6/7 (STOP_BOLTZ_GPU67=${STOP_BOLTZ_GPU67:-0})"
    return 0
  fi
  echo "Stopping Boltz workers on GPU 6/7 (avoid OOM with MPNN)..."
  pkill -f "launch_scaffolds_mpnn500_1000_gpu67_parallel.sh --worker 500 6" 2>/dev/null || true
  pkill -f "launch_scaffolds_mpnn500_1000_gpu67_parallel.sh --worker 1000 7" 2>/dev/null || true
  sleep 2
  # 仍在 6/7 上的 boltz pipeline
  pkill -f "results_boltz_mpnn500.*--ids" 2>/dev/null || true
  pkill -f "results_boltz_mpnn1000.*--ids" 2>/dev/null || true
  sleep 2
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-mpnn1000_top50_resume_gpu67}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    stop_boltz_on_gpus
    _gpus="${GPUS[*]:-6 7}"
    _procs="${PROCS_PER_GPU:-2}"
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
STOP_BOLTZ_GPU67=0 GPUS=$(printf %q "$_gpus") PROCS_PER_GPU=$(printf %q "$_procs") \
bash $(printf %q "$SCRIPT_DIR/launch_top50_mpnn1000_resume_gpu67.sh"); \
echo ALL DONE top50 resume; exec bash"
    tmux new-session -d -s "$SESSION" -n "resume" "$inner"
    echo "tmux: $SESSION  GPUs=${_gpus}  procs/gpu=${_procs}"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    run_one_job "${2:?id}" "${3:?temp}" "${4:-6}"
    ;;
  "")
    stop_boltz_on_gpus
    assign_and_run
    build_pending
    echo "Remaining after run: ${#PENDING[@]}"
    ;;
  *)
    echo "usage: $0 [--tmux | --one ID TEMP GPU]" >&2
    exit 1
    ;;
esac
