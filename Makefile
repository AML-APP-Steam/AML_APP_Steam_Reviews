.PHONY: install format lint typecheck test check docs docs-build scrape clean-data

install:
	uv sync
	uv run pre-commit install

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

docs:
	uv run pdoc -d google steam_reviews

docs-build:
	uv run pdoc -d google steam_reviews -o build/docs

scrape:
	uv run steam-reviews scrape

clean-data:
	uv run steam-reviews clean
