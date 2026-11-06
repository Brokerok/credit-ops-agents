from datetime import date

from app import rules
from app.schemas import Applicant, Decision, ExtractedApplication, Vehicle

TODAY = date(2026, 11, 6)


def _app(**over) -> ExtractedApplication:
    base = dict(
        applicant=Applicant(full_name="Olena Kovalenko", monthly_income_uah=48_000),
        vehicle=Vehicle(make="Toyota", model="Camry", year=2019, mileage_km=95_000, has_title=True),
        requested_amount_uah=150_000,
        requested_term_months=12,
    )
    base.update(over)
    return ExtractedApplication(**base)


def test_clean_application_is_auto_approved():
    app = _app()
    res, value, ltv = rules.evaluate(app, TODAY)
    assert value > 0 and ltv < 60
    assert all(r.passed for r in res)
    assert rules.decide(app, res) == Decision.AUTO_APPROVED


def test_ltv_above_70_is_hard_decline():
    app = _app(requested_amount_uah=1_500_000)
    res, _, ltv = rules.evaluate(app, TODAY)
    assert ltv > 70
    assert any(r.rule == "max_ltv_hard" and not r.passed for r in res)
    assert rules.decide(app, res) == Decision.DECLINED


def test_old_vehicle_is_declined_and_middle_age_goes_to_review():
    old = _app(vehicle=Vehicle(make="Toyota", model="Corolla", year=2005, has_title=True))
    res, _, _ = rules.evaluate(old, TODAY)
    assert rules.decide(old, res) == Decision.DECLINED
    mid = _app(vehicle=Vehicle(make="Toyota", model="Corolla", year=2013, has_title=True), requested_amount_uah=60_000)
    res, _, _ = rules.evaluate(mid, TODAY)
    assert rules.decide(mid, res) == Decision.HUMAN_REVIEW


def test_missing_income_or_title_requests_info():
    app = _app(applicant=Applicant(full_name="A B"))
    res, _, _ = rules.evaluate(app, TODAY)
    assert rules.decide(app, res) == Decision.NEEDS_INFO
    app = _app(vehicle=Vehicle(make="Toyota", model="Camry", year=2019, has_title=None))
    res, _, _ = rules.evaluate(app, TODAY)
    assert rules.decide(app, res) == Decision.NEEDS_INFO


def test_large_amount_never_auto_approves():
    app = _app(
        requested_amount_uah=400_000,
        vehicle=Vehicle(make="BMW", model="X5", year=2023, has_title=True),
        applicant=Applicant(full_name="A B", monthly_income_uah=200_000),
    )
    res, _, _ = rules.evaluate(app, TODAY)
    assert all(r.passed for r in res)
    assert rules.decide(app, res) == Decision.HUMAN_REVIEW


def test_valuation_depreciates_with_age_and_mileage():
    new = rules.estimate_vehicle_value("Toyota", 2024, 20_000, TODAY)
    old = rules.estimate_vehicle_value("Toyota", 2016, 20_000, TODAY)
    worn = rules.estimate_vehicle_value("Toyota", 2016, 350_000, TODAY)
    assert new > old > worn > 0
