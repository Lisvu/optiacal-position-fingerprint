#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$PROJECT_DIR/.venv_mac/bin/activate"

export MPLCONFIGDIR="/private/tmp/luminalink_matplotlib"
export PYTHONPYCACHEPREFIX="/private/tmp/luminalink_pycache"
export XDG_CACHE_HOME="/private/tmp/luminalink_cache"

mkdir -p "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME/fontconfig"

echo "Activated LuminaLink experiment environment:"
echo "  Python: $(which python)"
echo "  MPLCONFIGDIR: $MPLCONFIGDIR"
echo "  PYTHONPYCACHEPREFIX: $PYTHONPYCACHEPREFIX"
echo "  XDG_CACHE_HOME: $XDG_CACHE_HOME"
