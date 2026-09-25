#!/usr/bin/env bash
# Run the full unit-test suite.
set -e
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJ/.venv"
if [ -f "$VENV/Scripts/python.exe" ]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
cd "$PROJ"
"$PY" -m pytest -q
