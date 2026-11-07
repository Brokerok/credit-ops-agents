"""Scripted stand-in for an LLM so the public demo runs with no API key and no GPU.

`LLM_MODEL=demo` plugs this into both agents through pydantic-ai's `FunctionModel`. It is *not* a language
model: intake is regex extraction (EN/UK), the risk explanation is a template over the engine's rule results.
Everything downstream (policy engine, routing, human review, CRM, audit, evals) is the real code path, which
is exactly what the demo is meant to show. `/health` reports `demo:scripted` so nobody mistakes it for an LLM.
"""

from __future__ import annotations

import re

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.rules import HARD_DECLINE, KNOWN_MAKES

_NUM = r"\d{1,3}(?:[ ,.]\d{3})+|\d+"  # 150,000 / 150 000 / 150.000 / 150000, never "2019, 95,000"

_CITIES = {
    "kyiv": "Kyiv", "київ": "Київ", "києві": "Київ", "lviv": "Lviv", "львів": "Львів", "львові": "Львів",
    "odesa": "Odesa", "одеса": "Одеса", "одесі": "Одеса", "kharkiv": "Kharkiv", "харків": "Харків",
    "dnipro": "Dnipro", "дніпро": "Дніпро",
}  # fmt: skip
_RULE_SECTION = {
    "amount_range": ("credit_policy", "CP-2 Loan amount and term"),
    "max_ltv": ("credit_policy", "CP-3.2 Maximum loan-to-value (LTV)"),
    "max_ltv_hard": ("credit_policy", "CP-3.2 Maximum loan-to-value (LTV)"),
    "declared_value_gap": ("credit_policy", "CP-3.1 Vehicle valuation"),
    "vehicle_age": ("credit_policy", "CP-4 Vehicle age and mileage"),
    "vehicle_age_hard": ("credit_policy", "CP-4 Vehicle age and mileage"),
    "mileage": ("credit_policy", "CP-4 Vehicle age and mileage"),
    "min_income": ("credit_policy", "CP-5 Affordability"),
    "title_confirmed": ("collateral_rules", "CR-1 Ownership"),
}


def _num(s: str) -> int | None:
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def _last_user_text(messages: list[ModelMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    return p.content
    return ""


def extract(text: str) -> dict:
    """Regex 'intake'. Good enough for the sample applications; anything it cannot find goes to missing_fields."""
    t = text
    low = t.lower()
    missing: list[str] = []

    name = re.search(
        r"(?i:my name is|i am|мене звати|меня зовут)\s+([A-ZА-ЯІЇЄҐ][\w'’-]+(?:\s+[A-ZА-ЯІЇЄҐ][\w'’-]+){0,2})", t
    )
    phone = re.search(r"\+?\d[\d\s\-()]{8,}\d", t)
    income = re.search(r"(?:salary|income|earn|зарплат\w*|дох[іо]д\w*|заробля\w*)\D{0,40}?(" + _NUM + r")", low)
    amount = re.search(r"(?:borrow|loan of|need|потрібн\w*|позик\w*|кредит\w*|взяти)\D{0,30}?(" + _NUM + r")", low)
    if not amount:
        amount = re.search(r"(" + _NUM + r")\s*(?:uah|грн|₴|hryv)", low)
    term = re.search(r"(\d{1,3})\s*(?:months?|місяц\w*|міс\b)", low)
    year = re.search(r"\b(19[89]\d|20[0-3]\d)\b", t)
    mileage = re.search(r"(" + _NUM + r")\s*(?:km|км|thousand km|тис\.?\s*км)", low)
    declared = re.search(r"(?:worth|valued?|оцін\w*|варт\w*|коштує)\D{0,30}?(" + _NUM + r")", low)
    make = next((m for m in KNOWN_MAKES if re.search(rf"\b{re.escape(m)}\b", low)), None)
    model = None
    if make:
        mm = re.search(rf"\b{re.escape(make)}\b\s+([A-Za-z0-9\-]+)", low)
        model = mm.group(1).upper() if mm and mm.group(1) not in {"car", "the", "and"} else None
    city = next((v for k, v in _CITIES.items() if re.search(rf"\b{k}\b", low)), None)
    has_title = True if re.search(r"registration|title|техпаспорт|свідоцтв|own it|власни", low) else None
    if has_title and re.search(
        r"\b(no|not|without|не маю|нема[єе])\b[^.]{0,20}(registration|title|техпаспорт|свідоцтв)", low
    ):
        has_title = False

    for cond, field in [
        (not name, "applicant.full_name"), (not income, "applicant.monthly_income_uah"), (not phone, "applicant.phone"),
        (not make, "vehicle.make"), (not model, "vehicle.model"), (not year, "vehicle.year"),
        (not mileage, "vehicle.mileage_km"), (has_title is None, "vehicle.has_title"),
        (not amount, "requested_amount_uah"), (not term, "requested_term_months"),
    ]:  # fmt: skip
        if cond:
            missing.append(field)

    return {
        "applicant": {
            "full_name": name.group(1).strip() if name else "Unknown applicant",
            "phone": phone.group(0).strip() if phone else None,
            "city": city,
            "monthly_income_uah": _num(income.group(1)) if income else None,
        },
        "vehicle": {
            "make": make.title() if make else "Unknown",
            "model": model or "Unknown",
            "year": int(year.group(1)) if year else 2000,
            "mileage_km": _num(mileage.group(1)) if mileage else None,
            "declared_value_uah": _num(declared.group(1)) if declared else None,
            "has_title": has_title,
        },
        "requested_amount_uah": (_num(amount.group(1)) or 1) if amount else 1,
        "requested_term_months": int(term.group(1)) if term else None,
        "purpose": None,
        "missing_fields": missing,
        "language": "uk" if re.search(r"[а-яіїєґ]", low) else "en",
    }


def explain(prompt: str) -> dict:
    """Template 'risk analyst': reads the engine's numbers and rule lines back out of the prompt."""
    engine = re.search(r"estimated_vehicle_value_uah=(\d+), ltv_percent=([\d.]+)", prompt)
    rules = re.findall(r"^- (\w+): (PASS|FAIL) — (.+)$", prompt, re.M)
    failed = [(r, d) for r, ok, d in rules if ok == "FAIL"]
    if any(r in HARD_DECLINE for r, _ in failed):
        rec = "decline"
    elif failed:
        rec = "review"
    else:
        rec = "approve"
    if failed:
        rationale = f"{len(failed)} of {len(rules)} policy rules failed: " + "; ".join(d for _, d in failed) + "."
        rationale += (
            " Hard limit breached, decline." if rec == "decline" else " Soft limits only, route to a credit officer."
        )
    else:
        rationale = f"All {len(rules)} policy rules pass; LTV and affordability are within automated-approval limits."
    cited = {_RULE_SECTION[r] for r, _ in (failed or [(r, d) for r, _, d in rules]) if r in _RULE_SECTION}
    context = dict(re.findall(r"\[(\w+) \| ([^\]]+)\]\n", prompt))  # doc -> section blocks present in the prompt
    citations = []
    for doc, section in sorted(cited):
        quote = ""
        m = re.search(rf"\[{doc} \| {re.escape(section)}\]\n(.+?)(?:\n\n|\Z)", prompt, re.S)
        if m:
            quote = m.group(1).strip().split(". ")[0][:200]
        citations.append({"doc": doc, "section": section, "quote": quote or f"see {section}"})
    del context
    return {
        "estimated_vehicle_value_uah": int(engine.group(1)) if engine else 0,
        "ltv_percent": float(engine.group(2)) if engine else 0.0,
        "rule_results": [],
        "recommendation": rec,
        "rationale": rationale[:1500],
        "citations": citations,
    }


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    text = _last_user_text(messages)
    args = explain(text) if text.startswith("APPLICATION:") else extract(text)
    if info.output_tools:
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])
    import json

    return ModelResponse(parts=[TextPart(json.dumps(args))])


def build_demo_model() -> FunctionModel:
    return FunctionModel(_respond, model_name="scripted")
