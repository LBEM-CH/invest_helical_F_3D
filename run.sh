#!/usr/bin/env bash
# Launch invest_helical_F_3D with the project .venv, adding the local
# libxcb-cursor.so.0 shim that PyQt6 >= 6.5's xcb plugin needs (installed into
# ~/.local/lib without root). Usage:  ./run.sh <folder-or-star> [options]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
exec "$here/.venv/bin/python" "$here/invest_helical_F_3D.py" "$@"
