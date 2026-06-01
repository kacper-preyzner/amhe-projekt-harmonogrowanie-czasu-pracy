#!/usr/bin/env bash
# Uruchamia testy jednostkowe. Argumenty sa przekazywane do pytest.
#   ./test.sh                 # wszystkie testy
#   ./test.sh -k repair -v    # tylko wybrane, gadatliwie
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

echo ">> uv run pytest $*"
uv run pytest "$@"
