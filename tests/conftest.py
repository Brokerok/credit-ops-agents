"""Tests run fully offline: TestModel instead of an LLM, hash embedder instead of an ONNX model, SQLite file."""

import os
from pathlib import Path

import pytest

os.environ.update(
    LLM_MODEL="test:test",
    EMBEDDING_MODEL="hash",
    DATABASE_URL="sqlite:///./test_credit_ops.db",
    PYDANTIC_AI_NO_BANNER="1",
)


@pytest.fixture(autouse=True, scope="session")
def _clean_db():
    p = Path("test_credit_ops.db")
    p.unlink(missing_ok=True)
    yield
    p.unlink(missing_ok=True)
