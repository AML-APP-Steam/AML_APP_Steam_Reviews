# Developer commands for the Steam reviews project.
#
# Run `make help` (or just `make`) to list every command with a short description.
# Descriptions shown by `make help` come from the `## ...` text after each target name,
# so when adding a new target, document it the same way.

.DEFAULT_GOAL := help

.PHONY: help install format lint typecheck test check docs docs-build build scrape clean-data

# Print every target that has a `## description`, aligned in two columns.
help: ## Show this list of commands and what each one does
	@echo "Usage: make <command>"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------------

# Create/update the virtual environment from uv.lock and register the git
# pre-commit hooks so formatting/linting run automatically before each commit.
# The dev group pulls in the `scrape` and `analysis` extras; add the heavy
# transformer models with `uv sync --extra sentiment`.
install: ## Install dependencies with uv and set up pre-commit hooks
	uv sync --locked
	uv run pre-commit install

# --- Code quality ------------------------------------------------------------

# Rewrite files in place: apply ruff formatting, then auto-fix lint issues.
format: ## Auto-format code and fix lint issues with ruff (modifies files)
	uv run ruff format .
	uv run ruff check --fix .

# Read-only checks (safe for CI): fail if there are lint errors or if any
# file is not formatted. Does not change any files.
lint: ## Check code style and formatting with ruff (no changes made)
	uv run ruff check .
	uv run ruff format --check .

# Static type checking; settings come from pyproject.toml.
typecheck: ## Run static type checking with mypy
	uv run mypy

# Run the test suite in tests/.
test: ## Run the test suite with pytest
	uv run pytest

# Convenience target: everything that should pass before opening a PR.
check: lint typecheck test ## Run lint, typecheck and tests together (pre-PR check)

# --- Documentation -----------------------------------------------------------

# Start a local pdoc server with live API docs (Google-style docstrings).
docs: ## Serve API documentation locally in the browser (pdoc)
	uv run pdoc -d google steam_review_analysis

# Render the API docs to static HTML files in artifacts/docs.
docs-build: ## Build static HTML API documentation into artifacts/docs
	uv run pdoc -d google steam_review_analysis -o artifacts/docs

# --- Packaging ---------------------------------------------------------------

# Build the installable wheel and the source archive (sdist) into dist/.
# Releases are built and published by CI when a version tag is pushed (see CONTRIBUTING.md).
build: ## Build the wheel and source archive into dist/
	uv build

# --- Data pipeline -----------------------------------------------------------

# Scrape Steam reviews for every game in configs/games.toml into data/raw (uses Chrome).
# For options (specific app IDs, --headless, limits...), run:
#   uv run steam-review-analysis scrape --help
scrape: ## Scrape Steam reviews for all configured games into data/raw
	uv run steam-review-analysis scrape

# Merge and clean the raw CSVs into a single processed dataset; game names
# come from configs/games.toml.
clean-data: ## Clean raw data into data/processed/reviews.csv
	uv run steam-review-analysis clean
