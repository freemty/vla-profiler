#!/bin/bash
# Run test suite with coverage
# Usage: bash scripts/run_tests.sh [EXTRA_ARGS]
#   e.g., bash scripts/run_tests.sh
#         bash scripts/run_tests.sh -k test_attention
#         bash scripts/run_tests.sh --verbose
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python -m pytest tests/ \
    --tb=short \
    -q \
    "$@"
