"""Vector stores behind one interface: in-memory (dev, tests) and PostgreSQL + pgvector (docker compose).

Role filtering happens inside the store query, not after retrieval, so a chunk the caller is not allowed to
read is never loaded into the process at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sqlalchemy import Engine, text

from app.rag.chunker import Chunk
from app.rag.embedder import Embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def reindex(self, chunks: list[Chunk]) -> int: ...

    def search(self, query: str, role: str, k: int = 4) -> list[Hit]: ...


class MemoryStore:
    def __init__(self, embedder: Embedder) -> None:
        self._emb = embedder
        self._chunks: list[Chunk] = []
        self._vecs = np.zeros((0, embedder.dim), dtype=np.float32)

    def reindex(self, chunks: list[Chunk]) -> int:
        self._chunks = list(chunks)
        self._vecs = self._emb.embed([c.content for c in chunks]) if chunks else np.zeros((0, self._emb.dim))
        return len(self._chunks)

    def search(self, query: str, role: str, k: int = 4) -> list[Hit]:
        if not self._chunks:
            return []
        q = self._emb.embed([query])[0]
        allowed = [i for i, c in enumerate(self._chunks) if role in c.roles]
        if not allowed:
            return []
        scores = self._vecs[allowed] @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self._chunks[allowed[i]], float(scores[i])) for i in order]


class PgVectorStore:
    """Table policy_chunks(doc, section, roles text[], content, embedding vector(dim)); cosine distance via <=>."""

    def __init__(self, engine: Engine, embedder: Embedder) -> None:
        self._engine, self._emb = engine, embedder
        with engine.begin() as cx:
            cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            cx.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS policy_chunks ("
                    "id serial PRIMARY KEY, doc text NOT NULL, section text NOT NULL, roles text[] NOT NULL, "
                    f"content text NOT NULL, embedding vector({embedder.dim}) NOT NULL)"
                )
            )

    def reindex(self, chunks: list[Chunk]) -> int:
        vecs = self._emb.embed([c.content for c in chunks]) if chunks else []
        with self._engine.begin() as cx:
            cx.execute(text("TRUNCATE policy_chunks"))
            for c, v in zip(chunks, vecs, strict=True):
                cx.execute(
                    text(
                        "INSERT INTO policy_chunks (doc, section, roles, content, embedding) "
                        "VALUES (:doc, :section, :roles, :content, CAST(:embedding AS vector))"
                    ),
                    {
                        "doc": c.doc,
                        "section": c.section,
                        "roles": list(c.roles),
                        "content": c.content,
                        "embedding": "[" + ",".join(f"{x:.6f}" for x in v) + "]",
                    },
                )
        return len(chunks)

    def search(self, query: str, role: str, k: int = 4) -> list[Hit]:
        q = "[" + ",".join(f"{x:.6f}" for x in self._emb.embed([query])[0]) + "]"
        with self._engine.connect() as cx:
            rows = cx.execute(
                text(
                    "SELECT doc, section, roles, content, 1 - (embedding <=> CAST(:q AS vector)) AS score "
                    "FROM policy_chunks WHERE :role = ANY(roles) ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
                ),
                {"q": q, "role": role, "k": k},
            ).all()
        return [Hit(Chunk(r.doc, r.section, tuple(r.roles), r.content), float(r.score)) for r in rows]
