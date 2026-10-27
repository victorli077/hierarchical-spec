#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

MODEL_ROOT="/mnt/workspace/lijiasheng/huggingface"
MODEL_PATH="${MODEL_PATH:-${MODEL:-${MODEL_ROOT}/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-HierarchicalSpec}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-${SPECULATIVE_DRAFT_MODEL:-${MODEL_ROOT}/hub/models--Huang2020--Qwen3-4B-Domino-b16/snapshots/1a3f54218e6c42e943f13b1a19232682ea9810de}}"
SPECULATIVE_INTERMEDIATE_MODEL_PATH="${SPECULATIVE_INTERMEDIATE_MODEL_PATH:-${SPECULATIVE_INTERMEDIATE_MODEL:-${MODEL_ROOT}/hub/models--Qwen--Qwen3-4B-FP8/snapshots/96b30dc13593a244a5e59e84687309f53c375cfa}}"
SPECULATIVE_INTERMEDIATE_QUANTIZATION="${SPECULATIVE_INTERMEDIATE_QUANTIZATION:-fp8}"
SPECULATIVE_DRAFT_MODEL_QUANTIZATION="${SPECULATIVE_DRAFT_MODEL_QUANTIZATION:-}"
DTYPE="${DTYPE:-bfloat16}"

TASKS="${TASKS:-gsm8k:128,math500:128,humaneval:164,mbpp:128,livecodebench:128,alpaca:128}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32,64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:-1}"
CASCADE_ROUNDS="${CASCADE_ROUNDS:-3}"
DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
TP_SIZE="${TP_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
WARMUP_MAX_NEW_TOKENS="${WARMUP_MAX_NEW_TOKENS:-1024}"
SKIP_FIRST_REQUESTS="${SKIP_FIRST_REQUESTS:-1}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/sglang_hierarchical_$(date +%Y%m%d_%H%M%S)}"

if [[ "${SPECULATIVE_ALGORITHM,,}" != "hierarchicalspec" ]]; then
  echo "This three-way benchmark requires SPECULATIVE_ALGORITHM=HierarchicalSpec" >&2
  exit 2
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_DIR}/benchmark_three_way.py" ]]; then
  echo "Benchmark driver not found: ${SCRIPT_DIR}/benchmark_three_way.py" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${MODEL_ROOT}}"

quantization_lower="${SPECULATIVE_INTERMEDIATE_QUANTIZATION,,}"
if [[ "${quantization_lower}" == "int8" ]]; then
  SPECULATIVE_INTERMEDIATE_QUANTIZATION="w8a8_int8"
elif [[ "${quantization_lower}" == "awq" ]]; then
  # Compatible AWQ checkpoints are auto-converted to the Marlin kernel when
  # no override is supplied. Preserve that fast BF16-capable path when users
  # spell the checkpoint format as AWQ.
  SPECULATIVE_INTERMEDIATE_QUANTIZATION="awq_marlin"
else
  SPECULATIVE_INTERMEDIATE_QUANTIZATION="${quantization_lower}"
fi

args=(
  --target-model "${MODEL_PATH}"
  --draft-model "${SPECULATIVE_DRAFT_MODEL_PATH}"
  --intermediate-model "${SPECULATIVE_INTERMEDIATE_MODEL_PATH}"
  --intermediate-model-quantization "${SPECULATIVE_INTERMEDIATE_QUANTIZATION}"
  --dtype "${DTYPE}"
  --tasks "${TASKS}"
  --concurrencies "${CONCURRENCIES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --cascade-rounds "${CASCADE_ROUNDS}"
  --dflash-block-size "${DFLASH_BLOCK_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --tp-size "${TP_SIZE}"
  --page-size "${PAGE_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --warmup-max-new-tokens "${WARMUP_MAX_NEW_TOKENS}"
  --skip-first-requests "${SKIP_FIRST_REQUESTS}"
  --timeout-s "${TIMEOUT_S}"
  --output-dir "${OUT_DIR}"
)

if [[ -n "${SPECULATIVE_DRAFT_MODEL_QUANTIZATION}" ]]; then
  args+=(--speculative-draft-model-quantization "${SPECULATIVE_DRAFT_MODEL_QUANTIZATION}")
fi
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  args+=(--disable-radix-cache)
fi
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  args+=(--disable-cuda-graph)
fi
if [[ "${BATCH_REQUESTS:-0}" == "1" ]]; then
  args+=(--batch-requests)
fi

echo "Benchmarking A+B+C (HierarchicalSpec), C-only, and A+C (DFlash)"
echo "Target C: ${MODEL_PATH}"
echo "Drafter A: ${SPECULATIVE_DRAFT_MODEL_PATH}"
echo "Intermediate B: ${SPECULATIVE_INTERMEDIATE_MODEL_PATH} (${SPECULATIVE_INTERMEDIATE_QUANTIZATION})"
echo "Model dtype: ${DTYPE}"
echo "Tasks: ${TASKS}; concurrencies: ${CONCURRENCIES}"
echo "Output: ${OUT_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/benchmark_three_way.py" "${args[@]}" \
  2>&1 | tee "${OUT_DIR}/benchmark.log"
