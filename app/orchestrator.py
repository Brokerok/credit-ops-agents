"""The pipeline: intake agent -> policy engine -> risk agent -> decision -> CRM + audit.

Boundaries of autonomy are explicit: the LLM extracts and explains, code decides. Every step is audited,
every failure is caught and turned into a run status instead of a 500.
"""

from __future__ import annotations

import time
import uuid
from datetime import date

from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior
from sqlalchemy.orm import Session

from app import crm, rules
from app.agents.deps import AgentDeps
from app.agents.intake import build_intake_agent
from app.agents.risk import build_risk_agent
from app.config import settings
from app.db import Run, audit, get_engine
from app.logging import get_logger
from app.models_registry import ModelProfile, build_model
from app.rag.retriever import PolicyRetriever
from app.schemas import Decision, ExtractedApplication, PipelineResult, RiskAssessment

log = get_logger("pipeline")


class Pipeline:
    def __init__(self, profile: ModelProfile | None = None, retriever: PolicyRetriever | None = None) -> None:
        self.profile = profile or build_model()
        self.retriever = retriever or PolicyRetriever(get_engine())
        if retriever is None:
            self.retriever.reindex()
        self.intake = build_intake_agent(self.profile)
        self.risk = build_risk_agent(self.profile)

    async def run(self, text: str, source: str = "api", today: date | None = None) -> PipelineResult:
        run_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        usage_in = usage_out = 0
        errors: list[str] = []
        extracted: ExtractedApplication | None = None
        risk: RiskAssessment | None = None
        lead_id: str | None = None
        decision = Decision.NEEDS_INFO
        deps = AgentDeps(retriever=self.retriever, role="risk", run_id=run_id)
        engine = get_engine()
        with Session(engine) as session:
            session.add(Run(run_id=run_id, model=self.profile.spec, source=source))
            audit(session, run_id, "received", {"chars": len(text), "source": source})
            session.commit()

        try:
            # 1. intake: text -> structured facts
            r1 = await self.intake.run(text, deps=deps)
            extracted = r1.output
            usage_in += r1.usage.input_tokens or 0
            usage_out += r1.usage.output_tokens or 0

            # 2. deterministic policy engine
            rule_results, value, ltv = rules.evaluate(extracted, today)
            decision = rules.decide(extracted, rule_results)

            # 3. risk agent explains + cites policy (pre-retrieved context for local models)
            context = ""
            if not self.profile.native_tools:
                hits = []
                for rr in rule_results:
                    hits.extend(self.retriever.search(rr.detail, role=deps.role, k=1))
                seen, uniq = set(), []
                for h in hits:
                    if h.chunk.section not in seen:
                        seen.add(h.chunk.section)
                        uniq.append(h)
                context = "\n\nPOLICY CONTEXT:\n" + self.retriever.format_context(uniq)
            prompt = (
                f"APPLICATION:\n{extracted.model_dump_json(indent=1)}\n\n"
                f"ENGINE: estimated_vehicle_value_uah={value}, ltv_percent={ltv}\n"
                f"RULE RESULTS:\n"
                + "\n".join(f"- {x.rule}: {'PASS' if x.passed else 'FAIL'} — {x.detail}" for x in rule_results)
                + context
            )
            r2 = await self.risk.run(prompt, deps=deps)
            risk = r2.output
            usage_in += r2.usage.input_tokens or 0
            usage_out += r2.usage.output_tokens or 0
            # engine is the authority: enforce its numbers and note any disagreement for the reviewer
            risk.estimated_vehicle_value_uah, risk.ltv_percent, risk.rule_results = value, ltv, rule_results
            expected = {Decision.AUTO_APPROVED: "approve", Decision.DECLINED: "decline"}.get(decision, "review")
            if risk.recommendation != expected:
                errors.append(f"llm_disagreement: model said {risk.recommendation}, engine decided {decision}")

            # 4. CRM
            with Session(engine) as session:
                lead_id = crm.upsert_lead(session, run_id, extracted, decision)
                audit(
                    session, run_id, "decision", {"decision": decision, "rules": [x.model_dump() for x in rule_results]}
                )
                session.commit()
            status = "done"
        except (AgentRunError, UnexpectedModelBehavior) as e:
            errors.append(f"model_error: {e}")
            decision = Decision.HUMAN_REVIEW
            status = "failed"
            log.error("pipeline_model_error", run_id=run_id, error=str(e))
        except Exception as e:  # noqa: BLE001  anything else is still a routed run, never a crash
            errors.append(f"internal_error: {type(e).__name__}: {e}")
            decision = Decision.HUMAN_REVIEW
            status = "failed"
            log.exception("pipeline_internal_error", run_id=run_id)

        result = PipelineResult(
            run_id=run_id,
            decision=decision,
            extracted=extracted,
            risk=risk,
            crm_lead_id=lead_id,
            errors=errors,
            model=self.profile.spec,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            input_tokens=usage_in,
            output_tokens=usage_out,
            cost_usd=self.profile.cost(usage_in, usage_out),
        )
        with Session(engine) as session:
            run = session.get(Run, run_id)
            if run:
                run.status, run.decision, run.result = status, decision.value, result.model_dump(mode="json")
            audit(session, run_id, "finished", {"status": status, "latency_ms": result.latency_ms, "errors": errors})
            session.commit()
        log.info(
            "pipeline_done",
            run_id=run_id,
            decision=decision.value,
            latency_ms=result.latency_ms,
            model=self.profile.spec,
        )
        return result


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
    settings.__init__()  # re-read env (used by tests)
