"""LangGraph orchestrator: the same lending pipeline as `app/orchestrator.py`, expressed as a state graph.

    START -> intake -> policy -+-> explain -+-> crm -> END
                               |            +-> human_review (interrupt) -> crm -> END
                               +-> crm (needs_info: nothing to explain yet)
    any model error -> END with decision=human_review and the error recorded

What the graph adds over the linear pipeline:
- explicit routing (`add_conditional_edges`) instead of if/else inside one function;
- durable state through a checkpointer, so a run can pause at `human_review` (LangGraph `interrupt`) and be
  resumed later by a credit officer with `Command(resume=...)`, keyed by `run_id` as the thread id;
- every node is a unit that can be tested, retried or swapped on its own.

The agents are the same Pydantic AI agents: the graph is orchestration, not a second agent stack.
"""

from __future__ import annotations

import operator
import time
import uuid
from datetime import date
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior
from sqlalchemy.orm import Session

from app import crm, rules
from app.agents.deps import AgentDeps
from app.agents.intake import build_intake_agent
from app.agents.risk import build_risk_agent
from app.db import Run, audit, get_engine
from app.logging import get_logger
from app.models_registry import ModelProfile, build_model
from app.orchestrator import build_risk_prompt, reconcile_risk, retrieve_policy_context
from app.rag.retriever import PolicyRetriever
from app.schemas import Decision, ExtractedApplication, PipelineResult, RiskAssessment, RuleResult

log = get_logger("graph")

MODEL_ERRORS = (AgentRunError, UnexpectedModelBehavior)


class LoanState(TypedDict, total=False):
    # input
    run_id: str
    text: str
    source: str
    today: date | None
    # produced along the way
    extracted: ExtractedApplication
    rule_results: list[RuleResult]
    value: int
    ltv: float
    decision: Decision
    risk: RiskAssessment
    lead_id: str
    status: str
    reviewer: dict
    # reducers: nodes append / add instead of overwriting
    errors: Annotated[list[str], operator.add]
    usage_in: Annotated[int, operator.add]
    usage_out: Annotated[int, operator.add]


class GraphPipeline:
    """Same public contract as `Pipeline` (profile, retriever, intake, risk, run) plus `resume`."""

    def __init__(self, profile: ModelProfile | None = None, retriever: PolicyRetriever | None = None) -> None:
        self.profile = profile or build_model()
        self.retriever = retriever or PolicyRetriever(get_engine())
        if retriever is None:
            self.retriever.reindex()
        self.intake = build_intake_agent(self.profile)
        self.risk = build_risk_agent(self.profile)
        # In-memory checkpoints: fine for one process / demo. Production: langgraph-checkpoint-postgres,
        # pointed at the same PostgreSQL as the CRM and audit tables.
        allowed = [("app.schemas", n) for n in ("ExtractedApplication", "RuleResult", "RiskAssessment", "Decision")]
        self.checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    # ---------------------------------------------------------------- nodes

    def _deps(self, state: LoanState) -> AgentDeps:
        return AgentDeps(retriever=self.retriever, role="risk", run_id=state["run_id"])

    async def node_intake(self, state: LoanState) -> dict:
        try:
            r = await self.intake.run(state["text"], deps=self._deps(state))
        except MODEL_ERRORS as e:
            return self._model_error(state, "intake", e)
        self._audit(state, "node:intake", {"missing_fields": r.output.missing_fields})
        return {"extracted": r.output, "usage_in": r.usage.input_tokens or 0, "usage_out": r.usage.output_tokens or 0}

    def node_policy(self, state: LoanState) -> dict:
        rule_results, value, ltv = rules.evaluate(state["extracted"], state.get("today"))
        decision = rules.decide(state["extracted"], rule_results)
        self._audit(state, "node:policy", {"decision": decision, "rules": [x.model_dump() for x in rule_results]})
        return {"rule_results": rule_results, "value": value, "ltv": ltv, "decision": decision}

    async def node_explain(self, state: LoanState) -> dict:
        context = ""
        if not self.profile.native_tools:
            context = retrieve_policy_context(self.retriever, state["rule_results"], "risk")
        prompt = build_risk_prompt(state["extracted"], state["value"], state["ltv"], state["rule_results"], context)
        try:
            r = await self.risk.run(prompt, deps=self._deps(state))
        except MODEL_ERRORS as e:
            return self._model_error(state, "explain", e)
        risk = r.output
        disagreement = reconcile_risk(risk, state["value"], state["ltv"], state["rule_results"], state["decision"])
        self._audit(state, "node:explain", {"recommendation": risk.recommendation, "disagreement": disagreement})
        out: dict = {"risk": risk, "usage_in": r.usage.input_tokens or 0, "usage_out": r.usage.output_tokens or 0}
        if disagreement:
            out["errors"] = [disagreement]
        return out

    def node_human_review(self, state: LoanState) -> dict:
        """Pause here. The graph is checkpointed; `resume()` feeds the officer's verdict into `interrupt()`."""
        verdict = interrupt(
            {
                "run_id": state["run_id"],
                "question": "Engine routed this application to human review. Approve or decline?",
                "decision": state["decision"],
                "ltv_percent": state["ltv"],
                "failed_rules": [x.rule for x in state["rule_results"] if not x.passed],
                "rationale": state["risk"].rationale if state.get("risk") else None,
            }
        )
        decision = Decision.APPROVED if verdict.get("decision") == "approve" else Decision.DECLINED
        self._audit(state, "node:human_review", {"verdict": verdict, "decision": decision})
        return {"decision": decision, "reviewer": verdict}

    def node_crm(self, state: LoanState) -> dict:
        with Session(get_engine()) as session:
            lead_id = crm.upsert_lead(session, state["run_id"], state["extracted"], state["decision"])
            audit(session, state["run_id"], "node:crm", {"lead_id": lead_id, "decision": state["decision"]})
            session.commit()
        return {"lead_id": lead_id, "status": "done"}

    # ---------------------------------------------------------------- routing

    @staticmethod
    def after_intake(state: LoanState) -> str:
        return END if state.get("status") == "failed" else "policy"

    @staticmethod
    def after_policy(state: LoanState) -> str:
        # nothing to explain while facts are missing: ask the applicant first
        return "crm" if state["decision"] == Decision.NEEDS_INFO else "explain"

    @staticmethod
    def after_explain(state: LoanState) -> str:
        if state.get("status") == "failed":
            return END
        return "human_review" if state["decision"] == Decision.HUMAN_REVIEW else "crm"

    def _build(self) -> StateGraph:
        g = StateGraph(LoanState)
        g.add_node("intake", self.node_intake)
        g.add_node("policy", self.node_policy)
        g.add_node("explain", self.node_explain)
        g.add_node("human_review", self.node_human_review)
        g.add_node("crm", self.node_crm)
        g.add_edge(START, "intake")
        g.add_conditional_edges("intake", self.after_intake, {END: END, "policy": "policy"})
        g.add_conditional_edges("policy", self.after_policy, {"crm": "crm", "explain": "explain"})
        g.add_conditional_edges("explain", self.after_explain, {END: END, "human_review": "human_review", "crm": "crm"})
        g.add_edge("human_review", "crm")
        g.add_edge("crm", END)
        return g

    # ---------------------------------------------------------------- public API

    async def run(self, text: str, source: str = "api", today: date | None = None) -> PipelineResult:
        run_id = str(uuid.uuid4())
        with Session(get_engine()) as session:
            session.add(Run(run_id=run_id, model=self.profile.spec, source=source))
            audit(session, run_id, "received", {"chars": len(text), "source": source, "orchestrator": "langgraph"})
            session.commit()
        state: LoanState = {"run_id": run_id, "text": text, "source": source, "today": today, "errors": []}
        return await self._invoke(run_id, state)

    async def resume(self, run_id: str, verdict: dict) -> PipelineResult:
        """Continue a run paused at `human_review`. Raises ValueError if the run is not waiting."""
        snapshot = self.graph.get_state(self._config(run_id))
        if not snapshot.next or "human_review" not in snapshot.next:
            raise ValueError(f"run {run_id} is not waiting for review")
        return await self._invoke(run_id, Command(resume=verdict))

    async def _invoke(self, run_id: str, payload) -> PipelineResult:
        t0 = time.perf_counter()
        try:
            state = await self.graph.ainvoke(payload, config=self._config(run_id))
        except Exception as e:  # noqa: BLE001  unexpected failures become a routed run, never a 500
            log.exception("graph_internal_error", run_id=run_id)
            state = {"errors": [f"internal_error: {type(e).__name__}: {e}"], "decision": Decision.HUMAN_REVIEW}
            state["status"] = "failed"
        paused = "__interrupt__" in state
        status = "waiting_review" if paused else state.get("status", "failed")
        decision = state.get("decision", Decision.HUMAN_REVIEW)
        usage_in, usage_out = state.get("usage_in", 0), state.get("usage_out", 0)
        result = PipelineResult(
            run_id=run_id,
            decision=decision,
            extracted=state.get("extracted"),
            risk=state.get("risk"),
            crm_lead_id=state.get("lead_id"),
            errors=list(state.get("errors", [])),
            model=self.profile.spec,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            input_tokens=usage_in,
            output_tokens=usage_out,
            cost_usd=self.profile.cost(usage_in, usage_out),
        )
        with Session(get_engine()) as session:
            run = session.get(Run, run_id)
            if run:
                run.status, run.decision, run.result = status, decision.value, result.model_dump(mode="json")
            audit(session, run_id, "paused" if paused else "finished", {"status": status, "errors": result.errors})
            session.commit()
        log.info("graph_step_done", run_id=run_id, status=status, decision=decision.value, latency_ms=result.latency_ms)
        return result

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    @staticmethod
    def _audit(state: LoanState, event: str, payload: dict) -> None:
        with Session(get_engine()) as session:
            audit(session, state["run_id"], event, payload)
            session.commit()

    def _model_error(self, state: LoanState, node: str, e: Exception) -> dict:
        log.error("graph_model_error", run_id=state["run_id"], node=node, error=str(e))
        self._audit(state, f"node:{node}:error", {"error": str(e)[:500]})
        return {"errors": [f"model_error: {e}"], "decision": Decision.HUMAN_REVIEW, "status": "failed"}
