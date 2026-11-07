"""The scripted demo model must drive the real pipeline end to end without an LLM or API key."""

import pytest

from app.demo_model import explain, extract
from app.graph import GraphPipeline
from app.models_registry import build_model
from app.schemas import Decision, ExtractedApplication, RiskAssessment
from tests.test_pipeline import TODAY

EN = (
    "Hello, my name is Olena Kovalenko, phone +380 67 123 45 67, Kyiv. I would like to borrow 150,000 UAH for 12 "
    "months. The car is a Toyota Camry 2019, about 95,000 km, I own it and have the registration certificate. "
    "My salary is 48,000 UAH per month."
)
UK = (
    "Добрий день, мене звати Олена Коваленко, телефон +380 67 123 45 67, Київ. Хочу взяти кредит 150000 грн на "
    "12 місяців. Авто Toyota Camry 2019 року, пробіг 95000 км, техпаспорт на мене. Зарплата 48000 грн на місяць."
)


@pytest.mark.parametrize("text,lang", [(EN, "en"), (UK, "uk")])
def test_extract_reads_the_sample_applications(text, lang):
    app = ExtractedApplication.model_validate(extract(text))
    assert app.applicant.full_name.startswith("Olena") or app.applicant.full_name.startswith("Олена")
    assert app.applicant.monthly_income_uah == 48_000
    assert (app.vehicle.make, app.vehicle.model, app.vehicle.year) == ("Toyota", "CAMRY", 2019)
    assert app.vehicle.mileage_km == 95_000 and app.vehicle.has_title is True
    assert app.requested_amount_uah == 150_000 and app.requested_term_months == 12
    assert app.language == lang and app.missing_fields == []


def test_extract_does_not_glue_year_and_mileage():
    app = ExtractedApplication.model_validate(
        extract(
            "Hi, my name is Andrii Melnyk, Lviv. I need 500,000 UAH for 12 months. Car: Toyota Camry 2019, 95,000 km."
        )
    )
    assert (app.vehicle.year, app.vehicle.mileage_km, app.requested_amount_uah) == (2019, 95_000, 500_000)


def test_explain_follows_the_engine():
    prompt = (
        "APPLICATION:\n{}\n\nENGINE: estimated_vehicle_value_uah=722000, ltv_percent=69.3\nRULE RESULTS:\n"
        "- amount_range: PASS — CP-2 ok\n- max_ltv: FAIL — CP-3.2: LTV 69% > 60%\n"
        "- min_income: FAIL — CP-5: income low\n\n\nPOLICY CONTEXT:\n"
        "[credit_policy | CP-3.2 Maximum loan-to-value (LTV)]\nThe loan may not exceed 60% of value. More.\n"
    )
    risk = RiskAssessment.model_validate(explain(prompt))
    assert risk.recommendation == "review" and risk.ltv_percent == 69.3
    assert {c.section for c in risk.citations} == {"CP-3.2 Maximum loan-to-value (LTV)", "CP-5 Affordability"}
    assert risk.citations[0].quote.startswith("The loan may not exceed")


async def test_demo_model_runs_the_whole_graph():
    g = GraphPipeline(profile=build_model("demo:scripted"))
    ok = await g.run(EN, source="test", today=TODAY)
    assert ok.decision == Decision.AUTO_APPROVED and ok.crm_lead_id and ok.errors == []
    paused = await g.run(EN.replace("150,000", "500,000"), source="test", today=TODAY)
    assert paused.decision == Decision.HUMAN_REVIEW and paused.crm_lead_id is None
    done = await g.resume(paused.run_id, {"decision": "approve"})
    assert done.decision == Decision.APPROVED and done.crm_lead_id
    info = await g.run("Hello, I am Ivan. I want to borrow 200,000 UAH against my Skoda Octavia.", source="test")
    assert info.decision == Decision.NEEDS_INFO and "vehicle.year" in info.extracted.missing_fields
