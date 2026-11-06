.PHONY: install test lint run eval db

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

run:
	.venv/bin/uvicorn app.api:app --reload --port 8000

db:
	docker compose up -d db

eval:
	.venv/bin/python -m eval.run_eval --models $(MODELS)
