#!/bin/bash
# One-time environment setup on the PARAM Ganga LOGIN node (a few minutes).
# CentOS 7 ships Python 3.7 and glibc 2.17, so we use uv to fetch a standalone
# Python 3.11 and install the glibc-2.17-compatible pins.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements-glibc217.txt
.venv/bin/python -c "import sparse_dot_topn, lightgbm, pandas, rapidfuzz, anyascii; print('ENV OK', pandas.__version__, lightgbm.__version__)"
