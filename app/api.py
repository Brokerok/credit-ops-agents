"""FastAPI surface: submit an application (JSON text or an uploaded .txt/.pdf), read runs, reindex policy."""

from __future__ import annotations

import io
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Run, get_engine, list_runs
from app.logging import configure_logging
from app.orchestrator import get_pipeline
from app.schemas import PipelineResult, ReviewIn, RunRecord

configure_logging(json=settings.log_json)
app = FastAPI(
    title="credit-ops-agents",
    version="0.1.0",
    description="Agentic intake and risk pre-screening for vehicle-secured loans",
)


class ApplicationIn(BaseModel):
    text: str = Field(min_length=20, max_length=20_000, description="E-mail body, chat transcript or form dump")
    source: str = "api"


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {"status": "ok", "model": p.profile.spec, "vector_backend": p.retriever.backend}


@app.post("/applications", response_model=PipelineResult)
async def submit_application(body: ApplicationIn) -> PipelineResult:
    return await get_pipeline().run(body.text, source=body.source)


@app.post("/applications/upload", response_model=PipelineResult)
async def upload_application(file: Annotated[UploadFile, File()]) -> PipelineResult:
    raw = await file.read()
    if file.filename and file.filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages)
    else:
        text = raw.decode("utf-8", errors="replace")
    if len(text.strip()) < 20:
        raise HTTPException(422, "could not extract text from file")
    return await get_pipeline().run(text, source=f"upload:{file.filename}")


@app.get("/runs", response_model=list[RunRecord])
def runs(limit: int = 50) -> list[RunRecord]:
    with Session(get_engine()) as s:
        return [_to_record(r) for r in list_runs(s, limit)]


@app.get("/runs/{run_id}", response_model=RunRecord)
def run_detail(run_id: str) -> RunRecord:
    with Session(get_engine()) as s:
        r = s.get(Run, run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return _to_record(r)


@app.post("/runs/{run_id}/review", response_model=PipelineResult)
async def review_run(run_id: str, body: ReviewIn) -> PipelineResult:
    """Credit officer's verdict for a run paused at human review (ORCHESTRATOR=langgraph only)."""
    p = get_pipeline()
    if not hasattr(p, "resume"):
        raise HTTPException(409, "human-in-the-loop resume needs ORCHESTRATOR=langgraph")
    try:
        return await p.resume(run_id, body.model_dump())
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/policy/reindex")
def reindex() -> dict:
    return {"chunks": get_pipeline().retriever.reindex()}


def _to_record(r: Run) -> RunRecord:
    return RunRecord(
        run_id=r.run_id,
        created_at=r.created_at,
        status=r.status,
        decision=r.decision,
        model=r.model,
        source=r.source,
        result=PipelineResult.model_validate(r.result) if r.result else None,
    )
