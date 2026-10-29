#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SCRIPT_DIR}/outputs/c_kv_copy_${STAMP}}"
TIMING_INTERVAL="${HIERARCHICAL_TIMING_INTERVAL:-1}"

# Keep an invocation with no overrides short enough for a first correctness and
# timing probe. Set these variables before the script for a full benchmark.
export TASKS="${TASKS:-gsm8k:16}"
export CONCURRENCIES="${CONCURRENCIES:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
export WARMUP_MAX_NEW_TOKENS="${WARMUP_MAX_NEW_TOKENS:-128}"
export SKIP_FIRST_REQUESTS="${SKIP_FIRST_REQUESTS:-0}"
export SPECULATIVE_INTERMEDIATE_MODEL_PATH="${SPECULATIVE_INTERMEDIATE_MODEL_PATH:-${SPECULATIVE_INTERMEDIATE_MODEL:-${MODEL_PATH:-${MODEL:-Qwen/Qwen3-4B}}}}"
export SPECULATIVE_INTERMEDIATE_QUANTIZATION="${SPECULATIVE_INTERMEDIATE_QUANTIZATION:-unquant}"

mkdir -p "${EXPERIMENT_ROOT}"

run_case() {
  local name="$1"
  local copy_flag="$2"
  local out_dir="${EXPERIMENT_ROOT}/${name}"
  echo
  echo "===== ${name}: SGLANG_ENABLE_HIERARCHICAL_C_KV_COPY=${copy_flag} ====="
  env \
    OUT_DIR="${out_dir}" \
    BENCHMARK_MODES="abc" \
    HIERARCHICAL_C_KV_COPY="${copy_flag}" \
    HIERARCHICAL_TIMING="1" \
    HIERARCHICAL_TIMING_INTERVAL="${TIMING_INTERVAL}" \
    "${SCRIPT_DIR}/run_sglang_benchmark.sh"
}

run_case baseline 0
run_case c_kv_copy 1

"${PYTHON:-python}" "${SCRIPT_DIR}/summarize_c_kv_copy_experiment.py" \
  "${EXPERIMENT_ROOT}"

cat <<EOF

Experiment outputs:
  baseline:  ${EXPERIMENT_ROOT}/baseline/benchmark.log
  c_kv_copy: ${EXPERIMENT_ROOT}/c_kv_copy/benchmark.log
  summary:   ${EXPERIMENT_ROOT}/c_kv_copy_comparison.md

Compare the HierarchicalSpec sections for:
  - accept length / accept tokens
  - c_to_b_kv_copy_async and b_wait_for_c_to_b_kv_copy timing
  - b_round_* timing and total latency
EOF
