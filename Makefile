.PHONY: test lint build eval eval-live smoke clean

test:
	uv run pytest

lint:
	uv run ruff check .

build:
	uv build

# Pipeline validation against the simulated scorer (no credentials needed).
eval:
	uv run python evals/run_eval.py

# Real numbers against Cloudflare; needs CLEF_ACCOUNT_ID / CLEF_API_TOKEN.
eval-live:
	uv run python evals/run_eval.py --mode live

# Offline behaviour check of the CLI (no network).
smoke:
	uv run clef-compact --version

clean:
	rm -rf dist .pytest_cache .ruff_cache .coverage
