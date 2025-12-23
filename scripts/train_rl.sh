#!/bin/bash
# CSV Agent RL Training Script
#
# Prerequisites:
# 1. Install verifiers with RL extras: uv add 'verifiers[rl]'
# 2. Have episodes in test_fixtures/ or episodes/ directory
# 3. GPU available for training
#
# Usage:
#   ./scripts/train_rl.sh                    # Use default config
#   ./scripts/train_rl.sh --config custom    # Use custom config

set -e

CONFIG="${1:-configs/rl/csv-agent.toml}"

echo "=============================================="
echo "CSV Agent RL Training"
echo "=============================================="
echo "Config: $CONFIG"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Check for episodes
EPISODES_PATH=$(grep "episodes_path" "$CONFIG" | cut -d'"' -f2)
if [ ! -f "$EPISODES_PATH" ]; then
    echo "Warning: Episodes file not found: $EPISODES_PATH"
    echo "You may need to generate episodes first:"
    echo "  uv run python -m src.datagen.episode_gen"
    echo ""
fi

# Register the custom environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting RL training..."
echo ""

# Run verifiers RL trainer
uv run vf-rl @ "$CONFIG"
