.PHONY: help install dev lint format test pipeline clean

help:
	@echo "Targets: install dev lint format test pipeline clean"

install:  ## Install the package
	pip install -e .

dev:  ## Install with dev tools and git hooks
	pip install -e ".[dev]"
	pre-commit install

lint:  ## Run ruff, black --check, and mypy
	ruff check src tests
	black --check src tests
	mypy src

format:  ## Auto-fix lint issues and format
	ruff check --fix src tests
	black src tests

test:  ## Run the unit tests
	python -m unittest discover -s tests -v

pipeline:  ## Build the DB, train, and score in one step
	coffee-pipeline

clean:  ## Remove generated artifacts and caches
	rm -rf outputs coffee.db .pytest_cache .ruff_cache .mypy_cache
