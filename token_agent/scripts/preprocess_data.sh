#!/bin/bash
# Preprocess all active datasets into the unified mixed-benchmark parquet.
#
# Usage:
#   bash token_agent/scripts/preprocess_data.sh [--active_categories 0,1,2,3]
#
# Output: ~/data/token_agent_mixed/{train,test}.parquet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"

ACTIVE_CATS="${1:-0,1,2,3}"

python -m token_agent.data.preprocess_mixed_benchmark \
    --local_dir ~/data/token_agent_mixed \
    --active_categories "$ACTIVE_CATS"

echo "Done. Data saved to ~/data/token_agent_mixed/"
