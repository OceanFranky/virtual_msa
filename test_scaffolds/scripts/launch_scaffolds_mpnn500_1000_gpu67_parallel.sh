#!/usr/bin/env bash
# 全量 test_scaffolds：mpnn500 / mpnn1000，无 template + fix_all_if_no_x
# GPU 6 → mpnn500，GPU 7 → mpnn1000；每卡 2 进程并行（同 results_boltz_mpnn2000 目录风格）
#
#   bash launch_scaffolds_mpnn500_1000_gpu67_parallel.sh --tmux
#   bash launch_scaffolds_mpnn500_1000_gpu67_parallel.sh --worker 500 6
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
ROOT="${ROOT:-$REPO/test_scaffolds}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
NUM_CYCLES="${NUM_CYCLES:-5}"
SKIP_DONE="${SKIP_DONE:-1}"

set +u
[[ -f /data/wenhao/.bashrc ]] && source /data/wenhao/.bashrc 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate proteinhunter 2>/dev/null || true
set -u

PY="${PYTHON:-/data/wenhao/miniconda3/envs/proteinhunter/bin/python}"
BOLTZ="${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}"
CCD="${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}"

collect_ids() {
  local N="$1"
  COLLECTED_IDS=()
  for d in "$ROOT"/*/; do
    [[ -d "$d" ]] || continue
    local id
    id=$(basename "$d")
    [[ "$id" == scripts ]] && continue
    [[ -f "$d/${id}_mpnn${N}_top.npz" && -f "$d/${id}_mpnn${N}_top.a3m" ]] || continue
    if [[ "$SKIP_DONE" == "1" ]]; then
      local res="$REPO/results_boltz_mpnn${N}/$id/0_protein_hunter_design/run_0"
      if [[ -f "$res/${id}_run_0_predicted_cycle_${NUM_CYCLES}.pdb" ]]; then
        continue
      fi
    fi
    COLLECTED_IDS+=("$id")
  done
}

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/boltz_mpnn_gpu${gpu}.lock"
  exec 9>"$lock"
  flock 9
}

run_one() {
  local id="$1"
  local N="$2"
  local gpu="$3"
  _gpu_lock "$gpu"
  local RES="$REPO/results_boltz_mpnn${N}"
  local EVAL="$REPO/results_eval_boltz_mpnn${N}"
  mkdir -p "$RES/$id"
  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$RES/$id/0_protein_hunter_design" 2>/dev/null || true
  fi
  local log="$RES/${id}/boltz_run.log"
  {
    echo ""
    echo "===== $(date -Is)  id=$id  msa_n=$N  gpu=$gpu ====="
    echo "no_template  fix_all_if_no_x  num_designs=1  num_cycles=$NUM_CYCLES  msa_max_seqs=$N"
  } | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PIPE" run \
    --root "$ROOT" \
    --ids "$id" \
    --msa_n "$N" \
    --msa_max_seqs "$N" \
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
  echo "===== done $id msa_n=$N $(date -Is) =====" | tee -a "$log"
}

run_gpu_queue() {
  local N="$1"
  local gpu="$2"
  shift 2
  local ids=("$@")
  local batch=()
  for id in "${ids[@]}"; do
    batch+=("$id")
    if [[ ${#batch[@]} -ge $PROCS_PER_GPU ]]; then
      local pids=()
      for bid in "${batch[@]}"; do
        echo ">>> mpnn${N} gpu=$gpu start $bid"
        run_one "$bid" "$N" "$gpu" &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    local pids=()
    for bid in "${batch[@]}"; do
      echo ">>> mpnn${N} gpu=$gpu start $bid"
      run_one "$bid" "$N" "$gpu" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
  fi
}

run_msa_n_gpu() {
  local N="$1"
  local GPU="$2"
  collect_ids "$N"
  local RES="$REPO/results_boltz_mpnn${N}"
  mkdir -p "$RES"
  {
    echo ""
    echo "===== $(date -Is)  mpnn${N}  gpu=$GPU  procs=$PROCS_PER_GPU  n=${#COLLECTED_IDS[@]} ====="
  } | tee -a "$RES/batch_run.log"
  if [[ ${#COLLECTED_IDS[@]} -eq 0 ]]; then
    echo "nothing to run for mpnn${N}" | tee -a "$RES/batch_run.log"
    return 0
  fi
  run_gpu_queue "$N" "$GPU" "${COLLECTED_IDS[@]}"
  echo "===== ALL DONE mpnn${N} gpu=$GPU $(date -Is) =====" | tee -a "$RES/batch_run.log"
}

_tmux_worker_cmd() {
  local N="$1"
  local GPU="$2"
  printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
PROCS_PER_GPU=%q SKIP_DONE=%q NUM_CYCLES=%q CLEAN_PARTIAL=%q \
bash %q --worker %s %s; echo DONE mpnn%s gpu%s; exec bash' \
    "$PROCS_PER_GPU" "$SKIP_DONE" "$NUM_CYCLES" "${CLEAN_PARTIAL:-1}" \
    "$SCRIPT_DIR/launch_scaffolds_mpnn500_1000_gpu67_parallel.sh" "$N" "$GPU" "$N" "$GPU"
}

case "${1:-}" in
  --tmux)
    SESSION="${SESSION:-boltz_scaffolds_mpnn500_1000_gpu67}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    tmux new-session -d -s "$SESSION" -n "mpnn500-g6" "$(_tmux_worker_cmd 500 6)"
    tmux new-window -t "$SESSION" -n "mpnn1000-g7" "$(_tmux_worker_cmd 1000 7)"
    echo "tmux: $SESSION"
    echo "  GPU6: mpnn500  → results_boltz_mpnn500/   (${PROCS_PER_GPU} 进程/卡)"
    echo "  GPU7: mpnn1000 → results_boltz_mpnn1000/  (${PROCS_PER_GPU} 进程/卡)"
    echo "  num_cycles=$NUM_CYCLES  fix_all_if_no_x  no_template"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --worker)
    run_msa_n_gpu "${2:?msa_n}" "${3:?gpu}"
    ;;
  --one)
    run_one "${2:?id}" "${3:?msa_n}" "${4:?gpu}"
    ;;
  *)
    echo "usage: $0 [--tmux | --worker MSA_N GPU | --one ID MSA_N GPU]" >&2
    exit 1
    ;;
esac
