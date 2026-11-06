"""Builds the store from settings and exposes role-aware retrieval + a compact context formatter for prompts."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine

from app.config import settings
from app.rag.chunker import load_policy_dir
from app.rag.embedder import get_embedder
from app.rag.store import Hit, MemoryStore, PgVectorStore, VectorStore


class PolicyRetriever:
    def __init__(self, engine: Engine | None = None, policy_dir: str | None = None) -> None:
        emb = get_embedder(settings.embedding_model)
        use_pg = engine is not None and engine.url.get_backend_name().startswith("postgresql")
        self.store: VectorStore = PgVectorStore(engine, emb) if use_pg else MemoryStore(emb)  # type: ignore[arg-type]
        self.policy_dir = Path(policy_dir or settings.policy_dir)
        self.backend = "pgvector" if use_pg else "memory"

    def reindex(self) -> int:
        return self.store.reindex(load_policy_dir(self.policy_dir))

    def search(self, query: str, role: str, k: int = 4) -> list[Hit]:
        return self.store.search(query, role, k)

    @staticmethod
    def format_context(hits: list[Hit]) -> str:
        """What the agent sees. Each block carries doc + section so the model can cite them verbatim."""
        return "\n\n".join(f"[{h.chunk.doc} | {h.chunk.section}]\n{h.chunk.content}" for h in hits)
