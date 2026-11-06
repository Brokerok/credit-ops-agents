"""Intake agent: free text (e-mail body, chat transcript, PDF text) -> ExtractedApplication. Facts only."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.deps import AgentDeps
from app.models_registry import ModelProfile
from app.schemas import ExtractedApplication

INSTRUCTIONS = """You are the intake clerk of a lender that issues loans secured by the applicant's car.
Read the application text and extract the fields exactly as stated. Rules:
- Never guess. If a field is absent, leave it null and add its dotted path to `missing_fields`
  (e.g. "applicant.monthly_income_uah", "vehicle.mileage_km", "vehicle.has_title").
- Amounts: convert to integer UAH. If the applicant writes "$" or "USD", convert at 41 UAH per USD
  and note it in `purpose`.
- `vehicle.has_title` is true only if the applicant explicitly says they own the car / hold the
  registration certificate.
- `language` is the language of the text: uk, ru, en or other.
- Do not assess risk, do not comment, do not add fields that are not in the schema."""


def build_intake_agent(profile: ModelProfile) -> Agent[AgentDeps, ExtractedApplication]:
    return Agent(
        profile.model,
        output_type=profile.output_type(ExtractedApplication),
        instructions=INSTRUCTIONS,
        deps_type=AgentDeps,
        name="intake",
        retries=2,
    )
