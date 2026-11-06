"""Typed contracts between agents, the API and storage. Agents never exchange free text, only these models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    full_name: str = Field(description="Full name as written by the applicant")
    phone: str | None = Field(default=None, description="Phone in any format, or null")
    city: str | None = None
    monthly_income_uah: int | None = Field(default=None, ge=0, description="Declared monthly income in UAH, or null")


class Vehicle(BaseModel):
    make: str
    model: str
    year: int = Field(ge=1980, le=2030)
    mileage_km: int | None = Field(default=None, ge=0)
    declared_value_uah: int | None = Field(
        default=None, ge=0, description="Value the applicant believes the car is worth"
    )
    has_title: bool | None = Field(
        default=None, description="Applicant states they hold the vehicle registration/title"
    )


class ExtractedApplication(BaseModel):
    """Output of the intake agent: structured facts only, no judgement."""

    applicant: Applicant
    vehicle: Vehicle
    requested_amount_uah: int = Field(gt=0)
    requested_term_months: int | None = Field(default=None, ge=1, le=120)
    purpose: str | None = None
    missing_fields: list[str] = Field(default_factory=list, description="Fields the applicant did not provide")
    language: Literal["uk", "ru", "en", "other"] = "uk"


class PolicyCitation(BaseModel):
    doc: str
    section: str
    quote: str = Field(max_length=600)


class RuleResult(BaseModel):
    rule: str = Field(description="Short rule id, e.g. 'max_ltv', 'vehicle_age', 'min_income'")
    passed: bool
    detail: str = Field(max_length=300)


class RiskAssessment(BaseModel):
    """Output of the risk agent. Every rule verdict must cite the policy it comes from."""

    estimated_vehicle_value_uah: int = Field(ge=0)
    ltv_percent: float = Field(ge=0, description="requested_amount / estimated_vehicle_value * 100")
    rule_results: list[RuleResult] = Field(default_factory=list, description="Filled by the policy engine")
    recommendation: Literal["approve", "review", "decline"]
    rationale: str = Field(max_length=1500)
    citations: list[PolicyCitation] = Field(default_factory=list)


class Decision(StrEnum):
    AUTO_APPROVED = "auto_approved"
    HUMAN_REVIEW = "human_review"
    DECLINED = "declined"
    NEEDS_INFO = "needs_info"


class PipelineResult(BaseModel):
    run_id: str
    decision: Decision
    extracted: ExtractedApplication | None = None
    risk: RiskAssessment | None = None
    crm_lead_id: str | None = None
    errors: list[str] = Field(default_factory=list)
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    status: str
    decision: Decision | None
    model: str
    source: str
    result: PipelineResult | None
