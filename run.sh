#!/usr/bin/env bash
# Uruchamia eksperymenty. Bez argumentu = pelne 3 scenariusze (--all).
#   ./run.sh                      # pelne 3 scenariusze
#   ./run.sh --smoke              # szybki przebieg dymny (~10 s)
#   ./run.sh --scenario ablation  # pojedynczy scenariusz
#   ./run.sh --scenario vs_cpsat
#   ./run.sh --scenario disruption
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

ARGS=("$@")
if [ "${#ARGS[@]}" -eq 0 ]; then
  ARGS=(--all)
fi

echo ">> uv run python -m amhe.experiments.run ${ARGS[*]}"
uv run python -m amhe.experiments.run "${ARGS[@]}"
echo ">> Wyniki: results/*.csv | figury: figures/"
