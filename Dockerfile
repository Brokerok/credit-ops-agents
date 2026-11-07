FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip && pip install ".[obs]"
COPY . .

# Demo defaults (no API key, no GPU): scripted model, LangGraph orchestrator, SQLite in a writable dir.
# Override LLM_MODEL / DATABASE_URL / LOGFIRE_TOKEN in the environment for a real deployment.
ENV LLM_MODEL=demo:scripted ORCHESTRATOR=langgraph EMBEDDING_MODEL=hash \
    DATABASE_URL=sqlite:////tmp/credit_ops.db LOG_JSON=1 ENVIRONMENT=demo PORT=8000

EXPOSE 8000 7860
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
