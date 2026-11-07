# credit-ops-agents

[![CI](https://github.com/Brokerok/credit-ops-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/Brokerok/credit-ops-agents/actions/workflows/ci.yml)
[![Deploy to Render](https://img.shields.io/badge/deploy-Render%20free%20tier-46e3b7)](https://render.com/deploy?repo=https://github.com/Brokerok/credit-ops-agents)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Pydantic AI](https://img.shields.io/badge/agents-Pydantic%20AI-e92063)](https://ai.pydantic.dev/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1c3c3c)](https://langchain-ai.github.io/langgraph/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

Agentic intake and risk pre-screening for **vehicle-secured consumer loans**: an applicant's e-mail, chat
transcript or PDF goes in; a structured application, a policy-checked decision, a CRM lead and an audit trail
come out. Built to show how LLM agents fit inside a regulated lending workflow without being allowed to
make up the rules.

```
e-mail / PDF ──▶ intake agent ──▶ policy engine ──▶ risk agent ──▶ decision ──▶ CRM + audit
                 (LLM, extract)   (code, decide)   (LLM, explain   (code)       (DB, webhook)
                                                    + cite policy)
```

## Live demo

Run it locally in one command (below) or deploy it with the Render button above; the UI at `/` lets you paste an
application (EN or UK), watch it go through intake → policy → explanation, and approve or decline a run that paused
for human review. The hosted demo runs `LLM_MODEL=demo:scripted`: a regex
intake and a templated explainer stand in for the LLM so the demo needs no API key and no GPU. Everything else — policy
engine, LangGraph routing and interrupt, CRM hand-off, audit log, cost accounting — is the real code path. Point
`LLM_MODEL` at `anthropic:…`, `openai:…` or `ollama:…` and the same UI runs on a real model.

## What it demonstrates

| Requirement | Where |
|---|---|
| Multi-agent pipeline with typed hand-offs, retries, exception routing | `app/orchestrator.py`, `app/schemas.py` |
| Same pipeline as a **LangGraph** state graph: conditional routing, checkpointer, `interrupt` for **human-in-the-loop**, resume via API | `app/graph.py`, `POST /runs/{id}/review` |
| Agents built with **Pydantic AI** (tools, structured output, deps injection) | `app/agents/` |
| **RAG** over internal policy with **role-based access** enforced in the store query | `app/rag/`, `app/policy/*.md` (front matter `roles:`) |
| PostgreSQL + **pgvector** in prod, in-memory index for dev/tests | `app/rag/store.py`, `docker-compose.yml` |
| Deterministic rule engine: the LLM never sets thresholds or does arithmetic | `app/rules.py` |
| Explicit **boundaries of autonomy** (approve / review / decline / needs info; disagreement is logged, not obeyed) | `app/rules.py::decide`, `app/orchestrator.py` |
| **Model-agnostic**: Claude, OpenAI, or an open-weights model via Ollama, one config line | `app/models_registry.py` |
| **Evaluation harness**: decision accuracy, extraction accuracy, latency, tokens, cost per model | `eval/run_eval.py`, `eval/golden.jsonl` |
| FastAPI service, CRM adapter with idempotent upsert + webhook retry, audit log | `app/api.py`, `app/crm.py`, `app/db.py` |
| Structured logs with **PII masking**, secrets only from env | `app/logging.py`, `app/config.py` |
| **Tracing** with Pydantic Logfire / OpenTelemetry: request → graph node → agent run → model call → SQL, on by env var | `app/observability.py` |
| Scripted **demo mode** (no key, no GPU) that exercises the full pipeline; one-click deploy | `app/demo_model.py`, `Dockerfile`, `render.yaml` |
| Offline test-suite (pydantic-ai `TestModel`, hash embedder), CI, Docker | `tests/`, `.github/workflows/ci.yml` |

## Quick start

```bash
git clone https://github.com/Brokerok/credit-ops-agents && cd credit-ops-agents
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env            # pick a model: anthropic:… / openai:… / ollama:…

# local, no API key: Ollama + Llama 3.1 8B
ollama pull llama3.1:8b
.venv/bin/uvicorn app.api:app --port 8000

curl -s localhost:8000/applications -H 'content-type: application/json' \
  -d @- <<'EOF' | python -m json.tool
{"text": "Hello, my name is Olena Kovalenko, phone +380 67 123 45 67, Kyiv. I would like to borrow 150,000 UAH for 12 months. The car is a Toyota Camry 2019, about 95,000 km, I own it and have the registration certificate. My salary is 48,000 UAH per month."}
EOF
```

Response (abridged):

```json
{
  "decision": "auto_approved",
  "extracted": {"applicant": {"full_name": "Olena Kovalenko", "monthly_income_uah": 48000}, "vehicle": {"make": "Toyota", "year": 2019, "has_title": true}, "requested_amount_uah": 150000},
  "risk": {
    "estimated_vehicle_value_uah": 722000, "ltv_percent": 20.8,
    "rule_results": [{"rule": "max_ltv", "passed": true, "detail": "CP-3.2: LTV 21% vs 60% auto limit"}, "..."],
    "recommendation": "approve",
    "citations": [{"doc": "credit_policy", "section": "CP-3.2 Maximum loan-to-value (LTV)", "quote": "The loan amount may not exceed 60% of the estimated vehicle value"}]
  },
  "crm_lead_id": "…", "latency_ms": 4100, "input_tokens": 2310, "output_tokens": 410, "cost_usd": 0.0
}
```

Other endpoints: `POST /applications/upload` (.txt / .pdf), `GET /runs`, `GET /runs/{id}`, `POST /runs/{id}/review`
(credit officer's verdict for a paused run, LangGraph mode), `POST /policy/reindex`, `GET /health`.

### PostgreSQL + pgvector

```bash
docker compose up -d db
DATABASE_URL=postgresql+psycopg://credit:credit@127.0.0.1:5433/credit_ops .venv/bin/uvicorn app.api:app
# or everything in containers (API talks to Ollama on the host):
docker compose up --build
```

### Deploy the demo

```bash
docker build -t credit-ops-agents . && docker run -p 8000:8000 credit-ops-agents   # demo mode by default
```

`render.yaml` deploys the same image on Render's free tier; a Hugging Face Space with the Docker SDK works with the
same Dockerfile (it listens on `$PORT`, 7860 on Spaces). Set `LOGFIRE_TOKEN` there to get traces of the demo traffic.

### Tests and lint

```bash
make test   # 24 tests, fully offline: TestModel + hash embedder + SQLite
make lint
```

## Design decisions

**The LLM extracts and explains; code decides.** Every threshold (LTV 60/70 %, vehicle age 12/15 years,
income ≥ 2× payment, auto-approve cap 300 000 UAH) lives in `app/rules.py` and mirrors a numbered section of
`app/policy/credit_policy.md`. The risk agent receives the engine's verdicts and must cite the policy sections;
if its recommendation disagrees with the engine, the run is flagged `llm_disagreement` for the reviewer instead
of being overridden. This is what "boundary of autonomy" means in a lending process: the agent can route and
request information, it cannot change a limit or promise terms (CP-7).

**Two orchestrators, one contract.** `ORCHESTRATOR=native` runs the linear pipeline in `app/orchestrator.py`;
`ORCHESTRATOR=langgraph` runs the same steps as a LangGraph `StateGraph` (`app/graph.py`): nodes `intake → policy →
explain → crm`, conditional edges (`needs_info` skips the explainer, a model error ends the run as `human_review`), and a
`human_review` node that calls `interrupt()`. With a checkpointer the run pauses there, `GET /runs/{id}` shows
`waiting_review`, and `POST /runs/{id}/review {"decision": "approve"|"decline"}` resumes it with `Command(resume=…)`
using `run_id` as the thread id; the officer's verdict lands in the CRM stage and the audit log. The agents are the same
Pydantic AI agents in both modes: the graph is orchestration, not a second agent stack. The in-memory checkpointer is
for the demo; in production it would be `langgraph-checkpoint-postgres` on the same database as the CRM tables.

**Role-aware retrieval.** Policy documents declare who may read them (`roles: [risk, legal]`). The filter is
applied inside the vector query (`WHERE :role = ANY(roles)` in pgvector, pre-filter in the memory store), so a
sales agent physically never receives credit-policy chunks, and the risk agent never sees the sales playbook.
Chunks are whole policy sections, which keeps citations human-readable (`CP-3.2`) instead of token windows.

**Choosing a model per task.** `LLM_MODEL` selects provider and model; the eval harness runs the same golden
set through the whole pipeline and reports accuracy, latency and cost side by side:

```bash
.venv/bin/python -m eval.run_eval --models ollama:llama3.1:8b,openai:gpt-4.1-mini,anthropic:claude-sonnet-4-5
```

What I would expect from this comparison in production, and why the pipeline is shaped the way it is:

* Extraction (intake) is a structured, low-reasoning task: a small/cheap model or a local open-weights model is
  usually enough, and keeping PII on-prem is a plus for a lender. Local models get `PromptedOutput` (JSON in the
  prompt) because 8B-class models are unreliable at native tool calling; hosted models get native structured
  output and the `search_policy` tool.
* Explanation with citations is where model quality shows: Claude/GPT-class models cite the right sections and
  write a rationale a credit officer accepts; small models tend to paraphrase. Since the decision itself does
  not depend on the model, a weak explainer costs reviewer time, not correctness.
* The cost column is computed from token usage with a price table in `models_registry.py`, so the business
  trade-off is visible per run (`cost_usd` in every API response), not estimated in a spreadsheet.

Results of a local run are in [`eval/results.md`](eval/results.md).

**Observability.** `LOGFIRE_TOKEN=…` turns on Pydantic Logfire: `instrument_pydantic_ai()` records every agent run
with prompt, structured output, retries and token usage; `instrument_fastapi()` and `instrument_sqlalchemy()` wrap
the request and the SQL around it, so one run is one trace from HTTP to database. Logfire speaks OpenTelemetry, so
the same spans can go to Langfuse, Tempo or Honeycomb by adding a span processor. Without the token nothing is
imported and the service stays as light as in CI.

**Operational hygiene.** Every run is persisted with status, decision and the full result; every step writes an
audit event; model errors and unexpected exceptions become `human_review` runs with an `errors` list, never a
500. Logs are structured (JSON in Docker) and phone numbers / e-mails are masked before they reach the log
sink. Secrets are read from the environment only; `.env` is git-ignored.

## Layout

```
app/
  api.py             FastAPI endpoints
  orchestrator.py    linear pipeline, retries, audit, CRM hand-off (+ steps shared with graph.py)
  graph.py           LangGraph variant: state graph, routing, interrupt/resume for human review
  demo_model.py      scripted stand-in for the LLM (regex intake + templated explainer) for the public demo
  observability.py   Logfire / OpenTelemetry tracing, enabled by LOGFIRE_TOKEN
  static/index.html  demo UI served at /
  rules.py           deterministic policy engine (mirrors app/policy/*.md)
  schemas.py         typed contracts between steps
  models_registry.py provider-agnostic model factory + price table
  agents/            intake (extract) and risk (explain + cite) agents, Pydantic AI
  rag/               chunker (section-level), embedder (fastembed / hash), stores (memory, pgvector), retriever
  policy/            credit_policy.md, collateral_rules.md, sales_playbook.md  (roles in front matter)
  crm.py, db.py, logging.py, config.py
eval/                golden.jsonl + run_eval.py (model comparison)
tests/               offline tests (TestModel, hash embedder)
samples/             example applications (en / uk / incomplete)
```

## What is intentionally out of scope

Real vehicle valuation (the stub is a brand table with depreciation; production = market-data API), real CRM
(here: table + webhook), authentication on the API, and document OCR (PDF text extraction only).
