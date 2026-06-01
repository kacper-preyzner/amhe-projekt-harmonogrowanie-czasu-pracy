# Makefile — projekt AMHE nr 7 (harmonogramowanie call center)
# Wymaga zainstalowanego `uv` (https://docs.astral.sh/uv/).

.PHONY: help setup test smoke experiments report clean

help:
	@echo "Cele:"
	@echo "  make setup        - utworz srodowisko i zainstaluj zaleznosci (uv sync)"
	@echo "  make test         - uruchom testy jednostkowe (uv run pytest)"
	@echo "  make smoke        - szybki przebieg dymny eksperymentow"
	@echo "  make experiments  - pelne 3 scenariusze (CSV + wykresy + tabele)"
	@echo "  make report       - skompiluj raport LaTeX do PDF (wymaga pdflatex)"
	@echo "  make clean        - usun wygenerowane wyniki i artefakty"

setup:
	uv sync

test:
	uv run pytest

smoke:
	uv run python -m amhe.experiments.run --smoke

experiments:
	uv run python -m amhe.experiments.run --all

report:
	cd report && pdflatex -interaction=nonstopmode raport.tex \
		&& bibtex raport && pdflatex -interaction=nonstopmode raport.tex \
		&& pdflatex -interaction=nonstopmode raport.tex

clean:
	rm -f results/*.csv results/*.log
	rm -f figures/*.png figures/*.pdf
	rm -f report/figures/*.png report/figures/*.pdf report/figures/*.html
	rm -f report/tables/*.tex
	rm -f report/*.aux report/*.log report/*.out report/*.toc report/*.bbl report/*.blg
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
