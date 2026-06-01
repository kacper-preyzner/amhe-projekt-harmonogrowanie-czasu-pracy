#!/usr/bin/env bash
# Kompiluje raport LaTeX do report/raport.pdf (pdflatex + bibtex + 2x pdflatex).
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/report"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "BLAD: nie znaleziono 'pdflatex'. Zainstaluj dystrybucje TeX (np. TeX Live)." >&2
  exit 1
fi

echo ">> pdflatex (1/3)"
pdflatex -interaction=nonstopmode -halt-on-error raport.tex >/dev/null
echo ">> bibtex"
bibtex raport >/dev/null || true
echo ">> pdflatex (2/3)"
pdflatex -interaction=nonstopmode -halt-on-error raport.tex >/dev/null
echo ">> pdflatex (3/3)"
pdflatex -interaction=nonstopmode -halt-on-error raport.tex >/dev/null
echo ">> Gotowe: report/raport.pdf"
