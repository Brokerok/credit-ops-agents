"""LangGraph orchestrator: same offline setup as test_pipeline.py, plus the pause/resume path."""

import pytest
from pydantic_ai.models.test import TestModel

from app.graph import GraphPipeline
from app.schemas import Decision
from tests.test_pipeline import EXTRACTED, RISK, TODAY


@pytest.fixture
def graph() -> GraphPipeline:
    return GraphPipeline()


async def test_happy_path_runs_straight_through(graph: GraphPipeline):
    with (
        graph.intake.override(model=TestModel(custom_output_args=EXTRACTED)),
        graph.risk.override(model=TestModel(custom_output_args=RISK)),
    ):
        res = await graph.run("some application text long enough", source="test", today=TODAY)
    assert res.decision == Decision.AUTO_APPROVED
    assert res.crm_lead_id and res.errors == []
    assert res.risk and res.risk.estimated_vehicle_value_uah > 1  # engine numbers win, as in the linear pipeline


async def test_needs_info_skips_the_risk_agent(graph: GraphPipeline):
    incomplete = {**EXTRACTED, "applicant": {**EXTRACTED["applicant"], "monthly_income_uah": None}}
    with graph.intake.override(model=TestModel(custom_output_args=incomplete)):
        res = await graph.run("x" * 40, source="test", today=TODAY)  # risk agent not overridden: must not be called
    assert res.decision == Decision.NEEDS_INFO
    assert res.risk is None and res.crm_lead_id  # lead exists so sales can ask for the missing data


async def test_human_review_pauses_then_resumes_with_officer_verdict(graph: GraphPipeline):
    big = {**EXTRACTED, "requested_amount_uah": 500_000}  # LTV 69% and income too low: soft fails -> review
    review = {**RISK, "recommendation": "review"}
    with (
        graph.intake.override(model=TestModel(custom_output_args=big)),
        graph.risk.override(model=TestModel(custom_output_args=review)),
    ):
        paused = await graph.run("x" * 40, source="test", today=TODAY)
    assert paused.decision == Decision.HUMAN_REVIEW
    assert paused.crm_lead_id is None  # nothing reaches the CRM until a human decides
    snapshot = graph.graph.get_state({"configurable": {"thread_id": paused.run_id}})
    assert snapshot.next == ("human_review",)
    assert snapshot.interrupts[0].value["failed_rules"] == ["max_ltv", "min_income"]

    done = await graph.resume(paused.run_id, {"decision": "approve", "comment": "income verified by phone"})
    assert done.decision == Decision.APPROVED
    assert done.crm_lead_id
    with pytest.raises(ValueError):
        await graph.resume(paused.run_id, {"decision": "approve"})  # already finished


async def test_model_failure_ends_in_human_review_without_crm(graph: GraphPipeline):
    bad = {**EXTRACTED, "requested_amount_uah": -5}
    with graph.intake.override(model=TestModel(custom_output_args=bad)):
        res = await graph.run("x" * 40, source="test", today=TODAY)
    assert res.decision == Decision.HUMAN_REVIEW
    assert res.errors and res.errors[0].startswith("model_error")
    assert res.crm_lead_id is None
