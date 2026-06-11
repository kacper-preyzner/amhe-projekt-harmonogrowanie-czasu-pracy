# Makefile — projekt AMHE nr 7 (harmonogramowanie call center)
# Wymaga zainstalowanego `uv` (https://docs.astral.sh/uv/).

.PHONY: help setup test smoke experiments report clean

help:
	@echo "Cele:"
	@echo "  make setup        - utworz srodowisko i zainstaluj zaleznosci (uv sync)"
	@echo "  make test         - uruchom testy jednostkowe (uv run pytest)"
	@echo "  make smoke        - szybki przebieg dymny eksperymentow"
	@echo "  make experiments  - pelne 3 scenariusze (CSV + wykresy + tabele)"
	@echo "  make clean        - usun wygenerowane wyniki i artefakty"

setup:
	uv sync

test:
	uv run pytest

smoke:
	uv run python -m amhe.experiments.run --smoke

experiments:
	uv run python -m amhe.experiments.run --all

clean:
	rm -f results/*.csv results/*.log
	rm -f figures/*.png figures/*.pdf
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
