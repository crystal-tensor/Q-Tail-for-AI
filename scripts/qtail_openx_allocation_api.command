#!/bin/zsh
set -u

ROOT="/Users/avalok/work/Q-TAIL-MVP"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

cd "$ROOT" || exit 1
exec "$PYTHON" "$ROOT/tools/qtail_service_api.py" --port 8223
