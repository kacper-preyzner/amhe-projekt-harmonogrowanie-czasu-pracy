#!/usr/bin/env bash
# Pelny pipeline: srodowisko -> testy -> eksperymenty -> raport.
# Domyslnie pelny przebieg; flaga --smoke robi szybki przebieg dymny.
set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$HERE"

RUN_ARG="--all"
if [ "${1:-}" = "--smoke" ]; then
  RUN_ARG="--smoke"
fi

echo "===> [1/4] Srodowisko"
"$HERE/setup.sh"
echo "===> [2/4] Testy"
"$HERE/test.sh"
echo "===> [3/4] Eksperymenty ($RUN_ARG)"
"$HERE/run.sh" "$RUN_ARG"
echo "===> [4/4] Raport"
"$HERE/report.sh"
echo "===> Pipeline zakonczony."
