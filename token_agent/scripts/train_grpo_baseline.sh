#!/usr/bin/env bash
# Train the GRPO baseline on the mixed benchmark (no Token-Agent latent prefix).
#
# Usage:
#   bash token_agent/scripts/train_grpo_baseline.sh [model_path] [n_gpus]
#
# This is the standard GRPO training on the same mixed data.
# Compare results with Token-Agent to demonstrate the value of latent prefixes.

set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen3-30B-A3B}"
N_GPUS="${2:-8}"

echo "=== GRPO Baseline Training on Mixed Benchmark ==="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${N_GPUS}"

python -m token_agent.trainer.main_token_agent \
    --config-path ../config \
    --config-name grpo_baseline_trainer \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.n_gpus_per_node="${N_GPUS}"
