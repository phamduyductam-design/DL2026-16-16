.PHONY: check lint test structure

check: structure lint test

structure:
	python scripts/check_structure.py

lint:
	python -m ruff check .

test:
	python -m pytest -q
