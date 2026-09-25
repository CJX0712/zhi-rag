#!/usr/bin/env bash
# One-click environment setup for ZhiDa.
# Creates a local venv and installs the locked CORE dependencies (offline-default
# runtime + tests). No model download required.
set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJ/.venv"

if [ -f "$VENV/Scripts/python.exe" ]; then
  PY="$VENV/Scripts/python.exe"
else
  PY="$VENV/bin/python"
fi

if [ ! -x "$PY" ]; then
  echo "==> creating venv at $VENV"
  python3 -m venv "$VENV" || python -m venv "$VENV"
fi

echo "==> installing locked core dependencies"
"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install -r "$PROJ/requirements.lock.txt" -q
echo "==> setup complete. Next: bash scripts/run_demo.sh"
