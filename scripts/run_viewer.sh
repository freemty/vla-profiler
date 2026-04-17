#!/bin/bash
# Start the Flask viewer server
# Usage: bash scripts/run_viewer.sh [PORT]
#   e.g., bash scripts/run_viewer.sh 5001
set -e

PORT=${1:-5001}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting viewer on http://localhost:${PORT}"
python viewer/app.py
