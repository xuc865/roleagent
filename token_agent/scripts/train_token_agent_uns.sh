#!/bin/bash
# Launch Token-Agent training.
#
# Usage:
#   bash token_agent/scripts/train_token_agent.sh [HYDRA_OVERRIDES...]
#
# Examples:
#   # Default (mixed benchmark, Token-Agent enabled)
#   bash token_agent/scripts/train_token_agent.sh
#
#   # GRPO baseline (no latent prefix / triplet loss)
#   bash token_agent/scripts/train_token_agent.sh algorithm.token_agent.enable=false
#
#   # Quick debug with small batch
#   bash token_agent/scripts/train_token_agent.sh data.train_batch_size=4 data.val_batch_size=2 trainer.n_gpus_per_node=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"

python -m token_agent.trainer.main_token_agent "$@"
