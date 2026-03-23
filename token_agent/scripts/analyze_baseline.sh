#!/usr/bin/env bash
# Analyze failure modes of a trained baseline model on the mixed benchmark.
#
# Usage:
#   bash token_agent/scripts/analyze_baseline.sh <model_path> [output_dir] [max_samples]
#
# Example:
#   bash token_agent/scripts/analyze_baseline.sh ./checkpoints/grpo_step_500 ./analysis_grpo 1000
#
# If you already have evaluation results in JSONL format:
#   python -m token_agent.analysis.failure_mode_analysis \
#       --log_dir ./results/eval_records.jsonl --format jsonl

set -euo pipefail

MODEL_PATH="${1:?Error: provide model_path as first argument}"
OUTPUT_DIR="${2:-./analysis_output}"
MAX_SAMPLES="${3:-500}"
BACKEND="${4:-vllm}"

echo "=== Failure Mode Analysis ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Backend: ${BACKEND}"
echo ""

python -m token_agent.analysis.collect_eval_records \
    --model_path "${MODEL_PATH}" \
    --data_path ~/data/token_agent_mixed/test.parquet \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --backend "${BACKEND}" \
    --temperature 0.3 \
    --max_tokens 2048

echo ""
echo "Done. Report at: ${OUTPUT_DIR}/failure_mode_report.json"
