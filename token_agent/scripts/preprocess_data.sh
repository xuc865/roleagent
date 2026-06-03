#!/bin/bash
# Preprocess all active datasets into the unified mixed-benchmark parquet.
#
# Usage:
#   bash token_agent/scripts/preprocess_data.sh [--active_categories 0,1,2,3]
#
# Output: /mnt/workspace/wxc/roleagent/data/token_agent_mixed{train,test}.parquet
#
# ── 数据集采样比例接口 ──────────────────────────────────────────────────────────
# SAMPLE_RATIOS: JSON dict，键为数据集名，值为采样比例 (0, 1]。
# 未列出的数据集保留全量。空字符串 {} 表示全部保留。
#
# 各数据集原始数量（test.parquet 中的统计，供参考）：
#   popqa            14267 条
#   2wikimultihopqa  12576 条
#   triviaqa         11313 条
#   squad            10570 条
#   hotpotqa          7405 条
#   lighteval/MATH    5000 条
#   simpleqa          4326 条
#   nq                3610 条
#   musique           2417 条
#   openai/gsm8k      1319 条
#   bamboogle          125 条
#   aime_2024           30 条
#   aime_2025           30 条
#   math_dapo           28 条
#
# 示例（只保留 gsm8k 的 50%，MATH 的 80%）：
#   SAMPLE_RATIOS='{"openai/gsm8k": 0.5, "lighteval/MATH": 0.8}' \
#     bash token_agent/scripts/preprocess_data.sh
# ───────────────────────────────────────────────────────────────────────────────
SAMPLE_RATIOS="${SAMPLE_RATIOS:-{}}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source /mnt/workspace/wxc/roleagent/.setup_env.sh
export TOKEN_AGENT_DATA_DIR=/mnt/workspace/wxc/Agent/otherdata/
export PYTHONPATH=/mnt/workspace/wxc/roleagent:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=/mnt/workspace/wxc/Agent/models
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR=/mnt/workspace/wxc/roleagent/wandb
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CHECKPOINTS_DIR=/mnt/workspace/wxc/roleagent/checkpoints

cd "$ROOT_DIR"

ACTIVE_CATS="${1:-0,1,2,3}"

python -m token_agent.data.preprocess_mixed_benchmark \
    --local_dir /mnt/workspace/wxc/roleagent/data/token_agent_mixed \
    --active_categories "$ACTIVE_CATS" \
    --sample_ratios "$SAMPLE_RATIOS"

echo "Done. Data saved to /mnt/workspace/wxc/roleagent/data/token_agent_mixed"
