"""Dependencies injected into agent runs. Agents reach the outside world only through this object."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.retriever import PolicyRetriever


@dataclass
class AgentDeps:
    retriever: PolicyRetriever
    role: str  # "risk" | "sales" | "legal": decides which policy chunks the agent may read
    run_id: str
