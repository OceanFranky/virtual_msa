#!/usr/bin/env bash
# 8G52–8GBE：boltz_ph 无条件单体设计，3000 条虚拟 MSA，10 cycles，1 run
#
#   bash launch_boltz_ph_8G_series.sh           # 串行
#   bash launch_boltz_ph_8G_series.sh --tmux    # 5 窗并行（各绑一块 GPU）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
INPUT_ROOT="${INPUT_ROOT:-/data/wenhao/Protein-Hunter-server/test_scaffords_5.20}"
RESULTS_ROOT="${RESULTS_ROOT:-$INPUT_ROOT}"
PIPE="$REPO/pipeline/boltz_auto_pipeline.py"
PREP="$SCRIPT_DIR/prepare_scaffords_inputs.sh"

# tmux 非交互 shell 无 conda：显式激活 proteinhunter（bashrc 在 set -u 下可能报错）
set +u
if [[ -f /data/wenhao/.bashrc ]]; then
  # shellcheck source=/dev/null
  source /data/wenhao/.bashrc
fi
if command -v conda >/dev/null 2>&1; then
  conda activate proteinhunter 2>/dev/null || true
fi
set -u
_PH_PY="/data/wenhao/miniconda3/envs/proteinhunter/bin/python"
PYTHON="${PYTHON:-$_PH_PY}"
export PATH="/data/wenhao/miniconda3/envs/proteinhunter/bin:${PATH:-}"
export PYTHON

NUM_DESIGNS="${NUM_DESIGNS:-1}"
NUM_CYCLES="${NUM_CYCLES:-10}"
SKIP_EVAL="${SKIP_EVAL:-1}"

IDS=(8G52 8G53 8G64 8GAC 8GBE)
GPUS=(0 1 2 3 4)

bash "$PREP"

run_one() {
  local id="$1"
  local gpu="${2:-0}"
  local subdir="$INPUT_ROOT/$id"
  local seq
  seq=$(awk '/^>/{if(n++)exit} !/^>/{printf "%s",$0}' "$subdir/${id}.a3m" | tr -d '\n')
  local seqlen=${#seq}
  local min_len=$((seqlen - 20))
  local max_len=$((seqlen + 20))
  [[ "$min_len" -lt 50 ]] && min_len=50

  echo "===== $id  gpu=$gpu  len=$seqlen  cycles=$NUM_CYCLES  runs=$NUM_DESIGNS ====="
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$PIPE" run \
    --root "$INPUT_ROOT" \
    --results_root "$RESULTS_ROOT" \
    --num_designs "$NUM_DESIGNS" \
    --num_cycles "$NUM_CYCLES" \
    --gpu_id 0 \
    --mode unconditional \
    --unconditional_msa_mode precomputed \
    --use_msa true \
    --use_template true \
    --use_mask_template true \
    --binder_template_cif_chain_id A \
    --min_protein_length "$min_len" \
    --max_protein_length "$max_len" \
    --msa_max_seqs 3000 \
    --skip_eval "$([[ "$SKIP_EVAL" == 1 ]] && echo true || echo false)" \
    --boltz_model_path "${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}" \
    --ccd_path "${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}" \
    2>&1 | tee "$INPUT_ROOT/$id/boltz_ph_design.log"
}

# boltz_auto run_batch iterates ALL subdirs — run one ID at a time via temp root
run_one_isolated() {
  local id="$1"
  local gpu="${2:-0}"
  local subdir="$INPUT_ROOT/$id"
  local seq
  seq=$(awk '/^>/{if(n++)exit} !/^>/{printf "%s",$0}' "$subdir/${id}.a3m" | tr -d '\n')
  local seqlen=${#seq}
  local min_len=$((seqlen - 20))
  local max_len=$((seqlen + 20))
  [[ "$min_len" -lt 50 ]] && min_len=50

  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$RESULTS_ROOT/$id/0_protein_hunter_design"
  fi

  {
    echo "===== restart $(date -Is)  id=$id  gpu=$gpu  python=$PYTHON ====="
  } >"$RESULTS_ROOT/$id/boltz_ph_design.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO/boltz_ph/design.py" \
    --name "$id" \
    --mode unconditional \
    --unconditional_msa_mode precomputed \
    --binder_msa "$INPUT_ROOT/$id/${id}.npz" \
    --seq "$seq" \
    --binder_template_path "$INPUT_ROOT/$id/${id}_mask.pdb" \
    --binder_template_cif_chain_id A \
    --num_designs "$NUM_DESIGNS" \
    --num_cycles "$NUM_CYCLES" \
    --gpu_id 0 \
    --msa_mode single \
    --min_protein_length "$min_len" \
    --max_protein_length "$max_len" \
    --msa_max_seqs 3000 \
    --save_dir "$RESULTS_ROOT/$id" \
    --boltz_model_path "${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}" \
    --ccd_path "${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}" \
    2>&1 | tee -a "$RESULTS_ROOT/$id/boltz_ph_design.log"
}

MODE="${1:-}"
case "$MODE" in
  --tmux)
    SESSION="${SESSION:-boltz_8G_series_520}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    bash "$PREP"
    _tmux_cmd() {
      local id="$1" gpu="$2"
      printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; export PATH=%q:${PATH:-}; export PYTHON=%q; export CLEAN_PARTIAL=1; bash %q --one %q %s 2>&1 | tee -a %q; echo DONE %s; exec bash' \
        "/data/wenhao/miniconda3/envs/proteinhunter/bin" \
        "$_PH_PY" \
        "$SCRIPT_DIR/launch_boltz_ph_8G_series.sh" \
        "$id" "$gpu" \
        "$RESULTS_ROOT/$id/boltz_ph_design.log" \
        "$id"
    }
    first="${IDS[0]}"
    tmux new-session -d -s "$SESSION" -n "$first" "$(_tmux_cmd "$first" 0)"
    i=1
    for id in "${IDS[@]:1}"; do
      gpu="${GPUS[$i]:-${GPUS[0]}}"
      tmux new-window -t "$SESSION" -n "$id" "$(_tmux_cmd "$id" "$gpu")"
      i=$((i + 1))
    done
    echo "tmux: $SESSION  (GPU: 8G52=0 8G53=1 8G64=2 8GAC=3 8GBE=4)"
    echo "  attach: tmux attach -t $SESSION"
    exit 0
    ;;
  --one)
    bash "$PREP"
    run_one_isolated "${2:?id}" "${3:-0}"
    ;;
  "")
    bash "$PREP"
    for i in "${!IDS[@]}"; do
      run_one_isolated "${IDS[$i]}" "${GPUS[$i]:-0}"
    done
    ;;
  *)
    echo "usage: $0 [--tmux|--one ID GPU]" >&2
    exit 1
    ;;
esac

echo "all boltz_ph jobs finished under $RESULTS_ROOT"
