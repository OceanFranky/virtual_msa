#!/usr/bin/env bash
# Top50 × mpnn1000_t0.7 MSA：无 template，序列固定（fix_all_if_no_x）
#
#   bash launch_top50_boltz_mpnn1000_t0.7.sh --tmux
#   bash launch_top50_boltz_mpnn1000_t0.7.sh --one 8U0X 2
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"
CSV="${CSV:-$REPO/top50_from_top100_prefer_short.csv}"
TAG="${TAG:-t0.7}"
MSA_SUFFIX="_mpnn1000_${TAG}"

if [[ -n "${GPUS:-}" ]] && [[ "${GPUS}" != *"("* ]]; then
  read -ra GPUS <<< "${GPUS}"
else
  GPUS=(${GPUS[@]:-2 3 4 5})
fi
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
SKIP_DONE="${SKIP_DONE:-1}"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"
RES="$REPO/results_boltz_top50_mpnn1000_${TAG}"
EVAL="$REPO/results_eval_boltz_top50_mpnn1000_${TAG}"
NUM_CYCLES="${NUM_CYCLES:-10}"

load_ready_ids() {
  mapfile -t TOP50_IDS < <("$PY" - "$CSV" "$ROOT" "$MSA_SUFFIX" <<'PY'
import csv, sys
from pathlib import Path
csv_path, root, suffix = sys.argv[1:4]
ready = []
for row in csv.DictReader(open(csv_path, newline="")):
    pid = row["PDB_ID"].strip()
    a3m = Path(root) / pid / f"{pid}{suffix}.a3m"
    npz = Path(root) / pid / f"{pid}{suffix}.npz"
    if not (a3m.is_file() and npz.is_file()):
        continue
    n = sum(1 for ln in a3m.read_text().splitlines() if ln.startswith(">"))
    if n != 1001:
        continue
    ready.append(pid)
print("\n".join(ready))
PY
)
  echo "Top50 with valid t0.7 MSA: ${#TOP50_IDS[@]}"
}

is_done() {
  local id="$1"
  [[ -f "$RES/$id/0_protein_hunter_design/run_0/${id}_run_0_predicted_cycle_${NUM_CYCLES}.pdb" ]]
}

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/boltz_top50_t07_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local gpu="$2"
  _gpu_lock "$gpu"
  local a3m="$ROOT/$id/${id}${MSA_SUFFIX}.a3m"
  [[ -f "$a3m" ]] || { echo "missing MSA: $a3m" >&2; return 1; }
  if [[ "$SKIP_DONE" == "1" ]] && is_done "$id"; then
    echo "skip done $id"
    return 0
  fi
  mkdir -p "$RES/$id"
  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$RES/$id/0_protein_hunter_design" 2>/dev/null || true
  fi
  local log="$RES/${id}/boltz_run.log"
  {
    echo ""
    echo "===== $(date -Is)  id=$id  msa=${MSA_SUFFIX}  gpu=$gpu ====="
    echo "no_template  fix_all_if_no_x  num_designs=1  num_cycles=$NUM_CYCLES  msa_max_seqs=1000"
  } | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --ids "$id" \
    --msa_tag "$MSA_SUFFIX" \
    --msa_max_seqs 1000 \
    --results_root "$RES" \
    --eval_dir "$EVAL" \
    --mode unconditional \
    --unconditional_msa_mode precomputed \
    --use_msa true \
    --use_template false \
    --fix_all_if_no_x \
    --num_designs 1 \
    --num_cycles "$NUM_CYCLES" \
    --gpu_id 0 \
    --skip_eval \
    --boltz_model_path "$BOLTZ" \
    --ccd_path "$CCD" \
    2>&1 | tee -a "$log"
  echo "===== done $id ${TAG} $(date -Is) =====" | tee -a "$log"
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local ids=("$@")
  local batch=()
  for id in "${ids[@]}"; do
    batch+=("$id")
    if [[ ${#batch[@]} -ge $PROCS_PER_GPU ]]; then
      local pids=()
      for bid in "${batch[@]}"; do
        echo ">>> top50 t0.7 gpu=$gpu start $bid"
        run_one "$bid" "$gpu" &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    local pids=()
    for bid in "${batch[@]}"; do
      echo ">>> top50 t0.7 gpu=$gpu start $bid"
      run_one "$bid" "$gpu" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
}

assign_and_run() {
  load_ready_ids
  [[ ${#TOP50_IDS[@]} -gt 0 ]] || { echo "no ready IDs" >&2; exit 1; }
  declare -A QUEUE
  for g in "${GPUS[@]}"; do QUEUE[$g]=""; done
  local i=0
  for id in "${TOP50_IDS[@]}"; do
    [[ "$SKIP_DONE" == "1" ]] && is_done "$id" && continue
    local g="${GPUS[$((i % ${#GPUS[@]}))]}"
    QUEUE[$g]+="$id "
    i=$((i + 1))
  done
  echo "########## Boltz top50 mpnn1000_t0.7  GPUs=${GPUS[*]} procs/gpu=$PROCS_PER_GPU ##########"
  local pids=()
  for g in "${GPUS[@]}"; do
    # shellcheck disable=SC2206
    local ids=(${QUEUE[$g]})
    [[ ${#ids[@]} -eq 0 ]] && continue
    echo "GPU $g: ${#ids[@]} PDBs"
    run_gpu_queue "$g" "${ids[@]}" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
}

_tmux_cmd() {
  local gpu="$1"
  shift
  local ids=("$@")
  local inner_ids
  inner_ids=$(printf '%q ' "${ids[@]}")
  printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
GPUS=%q PROCS_PER_GPU=%q SKIP_DONE=%q \
bash %q --gpu-queue %s %s; echo DONE gpu%s; exec bash' \
    "$gpu" "$PROCS_PER_GPU" "$SKIP_DONE" \
    "$SCRIPT_DIR/launch_top50_boltz_mpnn1000_t0.7.sh" "$gpu" "$inner_ids" "$gpu"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-boltz_top50_mpnn1000_t07}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    _gpus="${GPUS[*]:-2 3 4 5}"
    _procs="${PROCS_PER_GPU:-1}"
    inner="set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
GPUS=$(printf %q "$_gpus") PROCS_PER_GPU=$(printf %q "$_procs") SKIP_DONE=$(printf %q "$SKIP_DONE") \
bash $(printf %q "$SCRIPT_DIR/launch_top50_boltz_mpnn1000_t0.7.sh"); \
echo ALL DONE top50 boltz t0.7; exec bash"
    tmux new-session -d -s "$SESSION" -n "t0.7" "$inner"
    echo "tmux: $SESSION  out=$RES"
    echo "  GPUs=${_gpus}  procs/gpu=${_procs}  (48 PDB with t0.7 MSA)"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --gpu-queue)
    run_gpu_queue "${2:?gpu}" "${@:3}"
    ;;
  --one)
    run_one "${2:?id}" "${3:-2}"
    ;;
  "")
    assign_and_run
    ;;
  *)
    echo "usage: $0 [--tmux | --gpu-queue GPU ID... | --one ID GPU]" >&2
    exit 1
    ;;
esac
