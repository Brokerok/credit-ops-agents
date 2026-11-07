"""Deterministic policy engine. The LLM extracts and explains; it does not do arithmetic or set thresholds.

Every threshold here mirrors a section in app/policy/*.md and the rule result carries that section id, so a
reviewer can trace each verdict to the policy text. Change the policy -> change this file in the same PR.
"""

from __future__ import annotations

from datetime import date

from app.config import settings
from app.schemas import Decision, ExtractedApplication, RuleResult

# Rough market values for the valuation stub (UAH, as of a ~3-year-old car). Production: market-data provider.
_BASE_VALUE_UAH = {
    "toyota": 1_100_000,
    "lexus": 1_500_000,
    "honda": 800_000,
    "mazda": 650_000,
    "nissan": 600_000,
    "volkswagen": 700_000,
    "skoda": 650_000,
    "audi": 1_200_000,
    "bmw": 1_400_000,
    "mercedes": 1_500_000,
    "mercedes-benz": 1_500_000,
    "ford": 550_000,
    "renault": 450_000,
    "hyundai": 650_000,
    "kia": 650_000,
    "tesla": 1_300_000,
    "chevrolet": 400_000,
    "daewoo": 150_000,
    "lada": 120_000,
    "vaz": 120_000,
    "zaz": 100_000,
}
KNOWN_MAKES = tuple(_BASE_VALUE_UAH)  # lower-case brand names the valuation stub knows
_DEFAULT_VALUE_UAH = 500_000

HARD_DECLINE = {"amount_range", "max_ltv_hard", "vehicle_age_hard", "salvage"}
CRITICAL_FIELDS = {"vehicle.year", "requested_amount_uah", "vehicle.make", "vehicle.model", "applicant.full_name"}


def estimate_vehicle_value(make: str, year: int, mileage_km: int | None, today: date | None = None) -> int:
    """Valuation stub: brand base price depreciated 10%/year from age 3, minus 5% per 100k km over 150k."""
    age = max(0, (today or date.today()).year - year)
    value = _BASE_VALUE_UAH.get(make.strip().lower(), _DEFAULT_VALUE_UAH) * (0.9 ** max(0, age - 3))
    if mileage_km and mileage_km > 150_000:
        value *= max(0.6, 1 - 0.05 * ((mileage_km - 150_000) / 100_000))
    return int(round(value, -3))


def monthly_payment(amount: int, term_months: int | None) -> float:
    term = term_months or 12
    return amount / term + amount * 0.03


def evaluate(app: ExtractedApplication, today: date | None = None) -> tuple[list[RuleResult], int, float]:
    """Returns (rule results, estimated value, LTV%). Pure function: same input, same output, no LLM."""
    today = today or date.today()
    value = estimate_vehicle_value(app.vehicle.make, app.vehicle.year, app.vehicle.mileage_km, today)
    ltv = app.requested_amount_uah / value * 100 if value else 999.0
    age = today.year - app.vehicle.year
    r: list[RuleResult] = []

    ok = 20_000 <= app.requested_amount_uah <= 2_000_000
    r.append(
        RuleResult(
            rule="amount_range",
            passed=ok,
            detail=f"CP-2: amount {app.requested_amount_uah:,} UAH, allowed 20,000-2,000,000",
        )
    )

    if ltv > 70:
        r.append(
            RuleResult(
                rule="max_ltv_hard",
                passed=False,
                detail=f"CP-3.2: LTV {ltv:.0f}% > 70% hard limit (value est. {value:,} UAH)",
            )
        )
    else:
        r.append(
            RuleResult(
                rule="max_ltv",
                passed=ltv <= 60,
                detail=f"CP-3.2: LTV {ltv:.0f}% vs 60% auto limit (value est. {value:,} UAH)",
            )
        )

    if age > 15:
        r.append(RuleResult(rule="vehicle_age_hard", passed=False, detail=f"CP-4: vehicle age {age} > 15 years"))
    else:
        r.append(
            RuleResult(rule="vehicle_age", passed=age < 12, detail=f"CP-4: vehicle age {age} years (review if 12-15)")
        )

    if app.vehicle.mileage_km is not None:
        r.append(
            RuleResult(
                rule="mileage",
                passed=app.vehicle.mileage_km <= 300_000,
                detail=f"CP-4: mileage {app.vehicle.mileage_km:,} km (review above 300,000)",
            )
        )

    if app.vehicle.declared_value_uah:
        gap = app.vehicle.declared_value_uah / value - 1
        r.append(
            RuleResult(
                rule="declared_value_gap", passed=gap <= 0.25, detail=f"CP-3.1: declared value {gap:+.0%} vs estimate"
            )
        )

    pay = monthly_payment(app.requested_amount_uah, app.requested_term_months)
    if app.applicant.monthly_income_uah is None:
        r.append(
            RuleResult(
                rule="min_income", passed=False, detail=f"CP-5: income not declared; est. payment {pay:,.0f} UAH/month"
            )
        )
    else:
        r.append(
            RuleResult(
                rule="min_income",
                passed=app.applicant.monthly_income_uah >= 2 * pay,
                detail=f"CP-5: income {app.applicant.monthly_income_uah:,} vs 2x payment {2 * pay:,.0f} UAH",
            )
        )

    r.append(
        RuleResult(
            rule="title_confirmed",
            passed=app.vehicle.has_title is True,
            detail="CR-1: registration certificate " + ("confirmed" if app.vehicle.has_title else "not confirmed"),
        )
    )
    return r, value, round(ltv, 1)


def decide(app: ExtractedApplication, rules: list[RuleResult]) -> Decision:
    """CP-6/CP-7 decision authority, in code so the LLM cannot talk its way around it."""
    failed = {x.rule for x in rules if not x.passed}
    # Missing critical facts come first: a hard rule computed from a placeholder value is not a decline,
    # it is a question to the applicant (the schema forces e.g. a year, so an absent one is still "a number").
    missing = set(app.missing_fields) & CRITICAL_FIELDS
    if missing:
        return Decision.NEEDS_INFO
    if failed & HARD_DECLINE:
        return Decision.DECLINED
    if app.vehicle.has_title is not True or app.applicant.monthly_income_uah is None:
        return Decision.NEEDS_INFO
    if not failed and app.requested_amount_uah <= settings.auto_approve_max_amount_uah:
        return Decision.AUTO_APPROVED
    return Decision.HUMAN_REVIEW
