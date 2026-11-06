"""Pipeline tests with pydantic-ai's TestModel: no network, deterministic outputs, real engine + DB + CRM."""

from datetime import date

import pytest
from pydantic_ai.models.test import TestModel

from app.orchestrator import Pipeline
from app.schemas import Decision

TODAY = date(2026, 11, 6)

EXTRACTED = {
    "applicant": {
        "full_name": "Olena Kovalenko",
        "phone": "+380671234567",
        "city": "Kyiv",
        "monthly_income_uah": 48000,
    },
    "vehicle": {
        "make": "Toyota",
        "model": "Camry",
        "year": 2019,
        "mileage_km": 95000,
        "declared_value_uah": 850000,
        "has_title": True,
    },
    "requested_amount_uah": 150000,
    "requested_term_months": 12,
    "purpose": "renovation",
    "missing_fields": [],
    "language": "en",
}
RISK = {
    "estimated_vehicle_value_uah": 1,
    "ltv_percent": 1.0,
    "rule_results": [],
    "recommendation": "approve",
    "rationale": "All rules pass.",
    "citations": [{"doc": "credit_policy", "section": "CP-3.2", "quote": "LTV ≤ 60%"}],
}


@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline()


async def test_happy_path_auto_approves_and_creates_crm_lead(pipeline: Pipeline):
    with (
        pipeline.intake.override(model=TestModel(custom_output_args=EXTRACTED)),
        pipeline.risk.override(model=TestModel(custom_output_args=RISK)),
    ):
        res = await pipeline.run("some application text long enough", source="test", today=TODAY)
    assert res.decision == Decision.AUTO_APPROVED
    assert res.crm_lead_id
    assert res.risk and res.risk.ltv_percent < 60 and res.risk.estimated_vehicle_value_uah > 1  # engine numbers win
    assert res.errors == []


async def test_engine_overrules_model_and_flags_disagreement(pipeline: Pipeline):
    too_much = {**EXTRACTED, "requested_amount_uah": 1_500_000}
    with (
        pipeline.intake.override(model=TestModel(custom_output_args=too_much)),
        pipeline.risk.override(model=TestModel(custom_output_args=RISK)),
    ):  # model still says "approve"
        res = await pipeline.run("x" * 40, source="test", today=TODAY)
    assert res.decision == Decision.DECLINED
    assert any(e.startswith("llm_disagreement") for e in res.errors)


async def test_model_failure_routes_to_human_review_instead_of_crashing(pipeline: Pipeline):
    bad = {**EXTRACTED, "requested_amount_uah": -5}  # fails schema validation on every retry
    with pipeline.intake.override(model=TestModel(custom_output_args=bad)):
        res = await pipeline.run("x" * 40, source="test", today=TODAY)
    assert res.decision == Decision.HUMAN_REVIEW
    assert res.errors and "model_error" in res.errors[0]
