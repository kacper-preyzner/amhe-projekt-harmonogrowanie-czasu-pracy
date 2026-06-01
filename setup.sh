#!/usr/bin/env bash
# Tworzy srodowisko i instaluje zaleznosci (uv sync).
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if ! command -v uv >/dev/null 2>&1; then
  echo "BLAD: nie znaleziono 'uv'. Zainstaluj: https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo ">> uv sync (Python 3.12 + zaleznosci)"
uv sync
echo ">> Gotowe. Srodowisko w .venv/"
