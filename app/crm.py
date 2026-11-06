"""CRM adapter. In this demo the CRM is a table + optional outbound webhook; in production it is the same
interface over the real CRM API. Idempotent on run_id so a retried pipeline never creates duplicate leads."""

from __future__ import annotations

import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import CrmLead
from app.logging import get_logger
from app.schemas import Decision, ExtractedApplication

log = get_logger("crm")

_STAGE = {
    Decision.AUTO_APPROVED: "approved_pending_docs",
    Decision.HUMAN_REVIEW: "risk_review",
    Decision.NEEDS_INFO: "waiting_for_client",
    Decision.DECLINED: "declined",
}


def upsert_lead(session: Session, run_id: str, app: ExtractedApplication, decision: Decision) -> str:
    lead = session.scalar(select(CrmLead).where(CrmLead.run_id == run_id))
    if lead is None:
        lead = CrmLead(lead_id=str(uuid.uuid4()), run_id=run_id, full_name=app.applicant.full_name, payload={})
        session.add(lead)
    lead.phone = app.applicant.phone
    lead.stage = _STAGE[decision]
    lead.amount_uah = app.requested_amount_uah
    lead.payload = app.model_dump(mode="json")
    session.flush()
    _notify(lead.lead_id, lead.stage, run_id)
    return lead.lead_id


def _notify(lead_id: str, stage: str, run_id: str) -> None:
    if not settings.crm_webhook_url:
        return
    body = {"lead_id": lead_id, "stage": stage, "run_id": run_id}
    for attempt in range(3):
        try:
            httpx.post(settings.crm_webhook_url, json=body, timeout=5).raise_for_status()
            return
        except httpx.HTTPError as e:  # noqa: PERF203
            log.warning("crm_webhook_failed", attempt=attempt, error=str(e))
    log.error("crm_webhook_gave_up", lead_id=lead_id)
