#!/bin/bash
# Simple wrapper to run examples with uv
# Usage: ./run-examples.sh verify-hf-token

set -e

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_TOKEN | xargs)
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not found in .env or environment"
    echo ""
    echo "Please:"
    echo "  1. Copy .env.example to .env"
    echo "  2. Edit .env and add your HF_TOKEN"
    echo "  3. Run this script again"
    exit 1
fi

SCRIPT_NAME="${1:-verify-hf-token}"

# Run with uv
uv run python "examples/${SCRIPT_NAME}.py" "${@:2}"
