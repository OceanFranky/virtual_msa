#!/usr/bin/env bash
# Top10 难合成：boltz_ph + mpnn{N}_top MSA → test_scaffords_5.20_{N}
#   --seq = a3m 第一条 query（非全 X）
#
#   MSA_N=1000 bash launch_top10_hard_boltz_mpnn1500.sh --tmux-9   # → test_scaffords_5.20_1000
#   MSA_N=1500 bash launch_top10_hard_boltz_mpnn1500.sh --tmux-9   # → test_scaffords_5.20_1500
#   MSA_N=1000 bash launch_top10_hard_boltz_mpnn1500.sh --tmux-234  # GPU 2/3/4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-/data/wenhao/Protein-Hunter-server}"
MSA_N="${MSA_N:-1500}"
SCAFFORDS_ROOT="${SCAFFORDS_ROOT:-/data/wenhao/Protein-Hunter-server/test_scaffords_5.20_${MSA_N}}"
INPUT_ROOT="${INPUT_ROOT:-$SCAFFORDS_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-$INPUT_ROOT}"
PREP="$SCRIPT_DIR/prepare_scaffords_inputs.sh"
GPU="${GPU:-1}"

TAG="${TAG:-_mpnn${MSA_N}_top}"
MSA_MAX="${MSA_MAX:-$MSA_N}"
NUM_DESIGNS="${NUM_DESIGNS:-1}"
NUM_CYCLES="${NUM_CYCLES:-10}"

TOP10_9=(8U0X 8IQC 8BAQ 8Q8I 8BAU 8R2C 8SV7 8EO5 8BUX)

if [[ -n "${IDS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_IDS=($IDS)
else
  RUN_IDS=("${TOP10_9[@]}")
fi

# 9 蛋白 × 1500 MSA → GPU 5/6/7 各 3 个串行
QUEUE_GPU5=(8U0X 8BAQ 8EO5)
QUEUE_GPU6=(8IQC 8Q8I 8BAU)
QUEUE_GPU7=(8R2C 8SV7 8BUX)
QUEUE_GPU2=(8U0X 8BAQ 8EO5)
QUEUE_GPU3=(8IQC 8Q8I 8BAU)
QUEUE_GPU4=(8R2C 8SV7 8BUX)
# 9 蛋白摊到 GPU 0–7（GPU7 串行 2 个）
QUEUE_GPU0=(8U0X)
QUEUE_GPU1=(8IQC)
QUEUE_GPU8_BA=(8BAQ)
QUEUE_GPU8_QI=(8Q8I)
QUEUE_GPU8_BU=(8BAU)
QUEUE_GPU8_R2=(8R2C)
QUEUE_GPU8_SV=(8SV7)
QUEUE_GPU8_EO=(8EO5 8BUX)

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
export PYTHON TAG MSA_MAX NUM_DESIGNS NUM_CYCLES INPUT_ROOT RESULTS_ROOT REPO

_gpu_lock() {
  local gpu="$1"
  local lock="/tmp/boltz_ph_gpu${gpu}.lock"
  exec 9>"$lock"
  if ! flock -n 9; then
    echo "ERROR: GPU $gpu 已有任务在跑（lock=$lock），拒绝并行启动" >&2
    exit 1
  fi
}

run_one() {
  local id="$1"
  local gpu="${2:-5}"
  _gpu_lock "$gpu"
  local subdir="$INPUT_ROOT/$id"
  local a3m="$subdir/${id}.a3m"
  if [[ ! -f "$a3m" ]]; then
    echo "missing $a3m — run prep first (TAG=$TAG)" >&2
    exit 1
  fi
  # 设计链 = a3m 第一条序列（query 行，非 mask 全 X）
  local seq
  seq=$(awk '/^>/{if(n++)exit} !/^>/{printf "%s",$0}' "$a3m" | tr -d '\n')
  local seqlen=${#seq}
  if [[ -z "$seq" ]]; then
    echo "ERROR $id: empty design seq — check $a3m" >&2
    exit 1
  fi
  if [[ "$(echo -n "$seq" | tr -cd X | wc -c)" -eq "$seqlen" ]]; then
    if [[ "${ALLOW_QUERY_ALLX:-0}" != "1" ]]; then
      echo "ERROR $id: design seq all-X — set ALLOW_QUERY_ALLX=1 for query-X runs" >&2
      exit 1
    fi
    # query 全 X：长度用 fasta（与 mask 设计区一致）
    local fasta="$INPUT_ROOT/$id/${id}.fasta"
    if [[ -f "$fasta" ]]; then
      seqlen=$(awk '/^>/{next} {gsub(/[^A-Za-z]/,""); s+=length($0)} END{print s+0}' "$fasta")
    fi
    if [[ "$seqlen" -lt 1 ]]; then
      seqlen=${#seq}
    fi
    echo "  design_seq: all-X (${#seq} chars), seqlen_bounds from fasta=$seqlen"
  fi
  local min_len=$((seqlen - 20))
  local max_len=$((seqlen + 20))
  [[ "$min_len" -lt 50 ]] && min_len=50

  local npz="$INPUT_ROOT/$id/${id}.npz"
  if [[ ! -f "$npz" ]]; then
    echo "missing $npz" >&2
    exit 1
  fi

  mkdir -p "$RESULTS_ROOT/$id"
  if [[ "${CLEAN_PARTIAL:-1}" == "1" ]]; then
    rm -rf "$RESULTS_ROOT/$id/0_protein_hunter_design"
  fi

  {
    echo "===== $(date -Is)  id=$id  gpu=$gpu  msa=$MSA_MAX  seq_len=$seqlen  tag=$TAG  out=$RESULTS_ROOT ====="
    echo "design_seq_source: a3m_query  head=${seq:0:40}..."
  } >>"$RESULTS_ROOT/$id/boltz_ph_design.log"

  local tpl_args=()
  if [[ "${USE_BINDER_TEMPLATE:-1}" == "1" ]]; then
    tpl_args=(
      --binder_template_path "$INPUT_ROOT/$id/${id}_mask.pdb"
      --binder_template_cif_chain_id A
    )
  else
    echo "  binder_template: (none)"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO/boltz_ph/design.py" \
    --name "$id" \
    --mode unconditional \
    --unconditional_msa_mode precomputed \
    --binder_msa "$npz" \
    --seq "$seq" \
    "${tpl_args[@]}" \
    --num_designs "$NUM_DESIGNS" \
    --num_cycles "$NUM_CYCLES" \
    --gpu_id 0 \
    --msa_mode single \
    --min_protein_length "$min_len" \
    --max_protein_length "$max_len" \
    --msa_max_seqs "$MSA_MAX" \
    --save_dir "$RESULTS_ROOT/$id" \
    --boltz_model_path "${BOLTZ_MODEL:-/data/wenhao/.boltz/boltz2_conf.ckpt}" \
    --ccd_path "${CCD_PATH:-/data/wenhao/.boltz/mols_fixed_v2}" \
    2>&1 | tee -a "$RESULTS_ROOT/$id/boltz_ph_design.log"
}

run_queue() {
  local gpu="$1"
  shift
  local ids=("$@")
  for id in "${ids[@]}"; do
    echo ">>> queue gpu=$gpu  start $id"
    [[ "${SKIP_GPU_WAIT:-0}" != "1" ]] && wait_gpu_idle "$gpu"
    run_one "$id" "$gpu"
    echo ">>> queue gpu=$gpu  DONE $id"
  done
}

prep_ids() {
  local id_list=("$@")
  IDS="${id_list[*]}" TAG="$TAG" SRC="${SRC:-/data/wenhao/Protein-Hunter-server/test_scaffolds}" \
    DST="$INPUT_ROOT" bash "$PREP"
}

wait_gpu_idle() {
  local gpu="$1"
  local max_util="${MAX_GPU_UTIL:-15}"
  local u
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
  if [[ -n "$u" && "$u" -gt "$max_util" ]]; then
    echo "GPU $gpu 利用率 ${u}% > ${max_util}%，等待 60s..." >&2
    sleep 60
    wait_gpu_idle "$gpu"
  fi
}

_tmux_queue_cmd() {
  local gpu="$1"
  shift
  local ids=("$@")
  local parts=()
  for id in "${ids[@]}"; do
    parts+=("bash $(printf %q "$SCRIPT_DIR/launch_top10_hard_boltz_mpnn1500.sh") --one $(printf %q "$id") $gpu")
  done
  local chain
  chain=$(IFS='; '; echo "${parts[*]}")
  printf 'set +u; source /data/wenhao/.bashrc 2>/dev/null; conda activate proteinhunter 2>/dev/null; set -u; \
export PATH=/data/wenhao/miniconda3/envs/proteinhunter/bin:$PATH; export PYTHON=%q; \
export MSA_N=%q INPUT_ROOT=%q RESULTS_ROOT=%q TAG=%q MSA_MAX=%q ALLOW_QUERY_ALLX=%q USE_BINDER_TEMPLATE=%q CLEAN_PARTIAL=1 SKIP_GPU_WAIT=%q; %s; echo DONE gpu%s; exec bash' \
    "$_PH_PY" "$MSA_N" "$INPUT_ROOT" "$RESULTS_ROOT" "$TAG" "$MSA_MAX" "${ALLOW_QUERY_ALLX:-0}" "${USE_BINDER_TEMPLATE:-1}" "${SKIP_GPU_WAIT:-1}" "$chain" "$gpu"
}

MODE="${1:-}"
case "$MODE" in
  --tmux-17)
    SESSION="${SESSION:-boltz_top10_${MSA_N}_gpu17}"
    QUEUE_GPU1=(8U0X 8BAQ 8EO5 8BAU)
    QUEUE_GPU7=(8IQC 8Q8I 8R2C 8SV7 8BUX)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      [[ "${FORCE:-0}" == "1" ]] && tmux kill-session -t "$SESSION" || {
        echo "tmux 已存在: $SESSION — FORCE=1 重开" >&2; exit 1; }
    fi
    prep_ids "${TOP10_9[@]}"
    tmux new-session -d -s "$SESSION" -n "gpu1" "$(_tmux_queue_cmd 1 "${QUEUE_GPU1[@]}")"
    tmux new-window -t "$SESSION" -n "gpu7" "$(_tmux_queue_cmd 7 "${QUEUE_GPU7[@]}")"
    echo "tmux: $SESSION  out=$RESULTS_ROOT  MSA=$MSA_MAX"
    echo "  GPU1: ${QUEUE_GPU1[*]}"
    echo "  GPU7: ${QUEUE_GPU7[*]}"
    ;;
  --tmux-9)
    SESSION="${SESSION:-boltz_top10_${MSA_N}_gpu567}"
    # 避免多个 tmux session 同卡重复提交
    for old in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E "^boltz_top10_${MSA_N}_" || true); do
      if [[ "$old" != "$SESSION" ]]; then
        echo "关闭旧 session: $old" >&2
        tmux kill-session -t "$old" 2>/dev/null || true
      fi
    done
    pkill -f "scaffords_5.20_${MSA_N}" 2>/dev/null || true
    sleep 2
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    prep_ids "${TOP10_9[@]}"
    tmux new-session -d -s "$SESSION" -n "gpu5" \
      "$(_tmux_queue_cmd 5 "${QUEUE_GPU5[@]}")"
    tmux new-window -t "$SESSION" -n "gpu6" \
      "$(_tmux_queue_cmd 6 "${QUEUE_GPU6[@]}")"
    tmux new-window -t "$SESSION" -n "gpu7" \
      "$(_tmux_queue_cmd 7 "${QUEUE_GPU7[@]}")"
    echo "tmux: $SESSION"
    echo "  输出目录: $RESULTS_ROOT  (MSA_N=$MSA_N，不覆盖其他 _1000/_1500/_3000 目录)"
    echo "  MSA=$MSA_MAX  TAG=$TAG  --seq=a3m首条query"
    echo "  GPU5: ${QUEUE_GPU5[*]}"
    echo "  GPU6: ${QUEUE_GPU6[*]}"
    echo "  GPU7: ${QUEUE_GPU7[*]}"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --tmux-8gpu)
    SESSION="${SESSION:-boltz_top10_${MSA_N}_gpu08}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    prep_ids "${TOP10_9[@]}"
    tmux new-session -d -s "$SESSION" -n "gpu0" "$(_tmux_queue_cmd 0 "${QUEUE_GPU0[@]}")"
    tmux new-window -t "$SESSION" -n "gpu1" "$(_tmux_queue_cmd 1 "${QUEUE_GPU1[@]}")"
    tmux new-window -t "$SESSION" -n "gpu2" "$(_tmux_queue_cmd 2 "${QUEUE_GPU8_BA[@]}")"
    tmux new-window -t "$SESSION" -n "gpu3" "$(_tmux_queue_cmd 3 "${QUEUE_GPU8_QI[@]}")"
    tmux new-window -t "$SESSION" -n "gpu4" "$(_tmux_queue_cmd 4 "${QUEUE_GPU8_BU[@]}")"
    tmux new-window -t "$SESSION" -n "gpu5" "$(_tmux_queue_cmd 5 "${QUEUE_GPU8_R2[@]}")"
    tmux new-window -t "$SESSION" -n "gpu6" "$(_tmux_queue_cmd 6 "${QUEUE_GPU8_SV[@]}")"
    tmux new-window -t "$SESSION" -n "gpu7" "$(_tmux_queue_cmd 7 "${QUEUE_GPU8_EO[@]}")"
    echo "tmux: $SESSION  out=$RESULTS_ROOT  MSA=$MSA_MAX  TAG=$TAG"
    echo "  GPU0: ${QUEUE_GPU0[*]}  GPU1: ${QUEUE_GPU1[*]}  GPU2: ${QUEUE_GPU8_BA[*]}"
    echo "  GPU3: ${QUEUE_GPU8_QI[*]}  GPU4: ${QUEUE_GPU8_BU[*]}  GPU5: ${QUEUE_GPU8_R2[*]}"
    echo "  GPU6: ${QUEUE_GPU8_SV[*]}  GPU7: ${QUEUE_GPU8_EO[*]} (串行)"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --tmux-234)
    SESSION="${SESSION:-boltz_top10_${MSA_N}_gpu234}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    prep_ids "${TOP10_9[@]}"
    tmux new-session -d -s "$SESSION" -n "gpu2" \
      "$(_tmux_queue_cmd 2 "${QUEUE_GPU2[@]}")"
    tmux new-window -t "$SESSION" -n "gpu3" \
      "$(_tmux_queue_cmd 3 "${QUEUE_GPU3[@]}")"
    tmux new-window -t "$SESSION" -n "gpu4" \
      "$(_tmux_queue_cmd 4 "${QUEUE_GPU4[@]}")"
    echo "tmux: $SESSION"
    echo "  输出: $RESULTS_ROOT  MSA=$MSA_MAX  TAG=$TAG"
    echo "  GPU2: ${QUEUE_GPU2[*]}"
    echo "  GPU3: ${QUEUE_GPU3[*]}"
    echo "  GPU4: ${QUEUE_GPU4[*]}"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --tmux-serial)
    SESSION="${SESSION:-boltz_top10_batch2_gpu${GPU}}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    prep_ids "${RUN_IDS[@]}"
    tmux new-session -d -s "$SESSION" -n "gpu${GPU}" \
      "$(_tmux_queue_cmd "$GPU" "${RUN_IDS[@]}")"
    echo "tmux: $SESSION  GPU=$GPU  串行: ${RUN_IDS[*]}"
    echo "  (GPU 利用率>${MAX_GPU_UTIL:-15}% 时自动等待，避免抢卡)"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --tmux)
    SESSION="${SESSION:-boltz_top10_mpnn1500}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      if [[ "${FORCE:-0}" == "1" ]]; then
        tmux kill-session -t "$SESSION"
      else
        echo "tmux 已存在: $SESSION — 使用 FORCE=1 重开" >&2
        exit 1
      fi
    fi
    prep_ids "${QUEUE_GPU5[@]}" "${QUEUE_GPU6[@]}" "${QUEUE_GPU7[@]}"
    tmux new-session -d -s "$SESSION" -n "gpu5" "$(_tmux_queue_cmd 5 "${QUEUE_GPU5[@]}")"
    tmux new-window -t "$SESSION" -n "gpu6" "$(_tmux_queue_cmd 6 "${QUEUE_GPU6[@]}")"
    tmux new-window -t "$SESSION" -n "gpu7" "$(_tmux_queue_cmd 7 "${QUEUE_GPU7[@]}")"
    echo "tmux: $SESSION"
    echo "  GPU5: ${QUEUE_GPU5[*]}"
    echo "  GPU6: ${QUEUE_GPU6[*]}"
    echo "  GPU7: ${QUEUE_GPU7[*]}"
    echo "  MSA=$MSA_MAX  TAG=$TAG  --seq=a3m首条query  out=$RESULTS_ROOT"
    echo "  attach: tmux attach -t $SESSION"
    ;;
  --one)
    prep_ids "${2:?id}"
    run_one "${2:?id}" "${3:-$GPU}"
    ;;
  --serial)
    prep_ids "${RUN_IDS[@]}"
    run_queue "$GPU" "${RUN_IDS[@]}"
    ;;
  "")
    prep_ids "${RUN_IDS[@]}"
    run_queue "$GPU" "${RUN_IDS[@]}"
    ;;
  *)
    echo "usage: MSA_N=1000 $0 [--tmux-8gpu|--tmux-17|--tmux-9|--tmux-234|--tmux-serial|--one ID GPU]" >&2
    exit 1
    ;;
esac
