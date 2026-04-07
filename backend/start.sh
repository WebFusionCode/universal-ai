#!/bin/bash
set -e

PORT=${PORT:-10000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

echo "Starting server on port $PORT"

exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
