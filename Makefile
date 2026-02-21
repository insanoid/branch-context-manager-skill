.PHONY: setup format lint type test check fix precommit

setup:
	uv sync --group dev

format:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy scripts tests

test:
	uv run pytest

check: lint type test

fix:
	uv run ruff check --fix .
	uv run ruff format .

precommit:
	uv run pre-commit run --all-files
