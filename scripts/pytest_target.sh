#!/usr/bin/env bash
# Run targeted pytest selections with the repo virtualenv.
# Use this when you want a quick single-file / single-test run without
# accidentally falling back to system python and missing dev extras.
#
# Examples:
#   scripts/pytest_target.sh tests/agent/test_auxiliary_client.py::TestAuxiliaryPoolAwareness::test_async_call_llm_retries_nous_after_401
#   scripts/pytest_target.sh tests/agent/test_context_compressor.py -k default_threshold
#   scripts/pytest_target.sh -k 'call_llm_retries_nous_after_401'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV=""
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  if [ -f "$candidate/bin/activate" ]; then
    VENV="$candidate"
    break
  fi
done

if [ -z "$VENV" ]; then
  echo "error: no virtualenv found in $REPO_ROOT/.venv or $REPO_ROOT/venv" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"
cd "$REPO_ROOT"

echo "▶ running targeted pytest via $PYTHON"
exec "$PYTHON" -m pytest -o "addopts=" "$@"
