"""Persistence: runs, CRM leads (stub of the real CRM), audit events. SQLite by default, PostgreSQL via DATABASE_URL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Engine, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    status: Mapped[str] = mapped_column(String(20), default="running")
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    model: Mapped[str] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(40))
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class CrmLead(Base):
    __tablename__ = "crm_leads"
    lead_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    stage: Mapped[str] = mapped_column(String(30))
    amount_uah: Mapped[int | None]
    payload: Mapped[dict] = mapped_column(JSON)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    event: Mapped[str] = mapped_column(String(60))
    payload: Mapped[str] = mapped_column(Text)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    engine = create_engine(settings.database_url, future=True)
    Base.metadata.create_all(engine)
    return engine


def audit(session: Session, run_id: str, event: str, payload: dict | str) -> None:
    session.add(
        AuditEvent(
            run_id=run_id,
            event=event,
            payload=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str),
        )
    )


def list_runs(session: Session, limit: int = 50) -> list[Run]:
    return list(session.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)))
