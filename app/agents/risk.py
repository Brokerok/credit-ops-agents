"""Risk agent: explains the deterministic rule results against the policy text and cites sections.

Two modes, chosen by the model profile:
- hosted models (Claude/OpenAI) get a `search_policy` tool and look things up themselves;
- small local models get the relevant policy sections pre-retrieved into the prompt (no tool calls to break).
Either way the recommendation cannot override the rule engine: the orchestrator records disagreement instead.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from app.agents.deps import AgentDeps
from app.models_registry import ModelProfile
from app.schemas import RiskAssessment

INSTRUCTIONS = """You are a credit risk analyst at a vehicle-secured lender.
You receive: the extracted application, the valuation and LTV computed by the policy engine, and the engine's
rule-by-rule results. Your job is to explain, not to re-decide:
- `rule_results`: leave empty; the engine attaches its own results after your answer.
- `recommendation`: approve only if every rule passed; decline if a hard rule failed (amount_range, max_ltv_hard,
  vehicle_age_hard); otherwise review.
- `rationale`: 2-5 sentences a credit officer can read in 20 seconds. Mention the decisive rules with numbers.
- `citations`: for each decisive rule cite the policy section it comes from, quoting at most one sentence.
  Only cite sections that appear in the provided policy context or tool results. Never invent sections.
- Copy `estimated_vehicle_value_uah` and `ltv_percent` from the engine as given."""


def build_risk_agent(profile: ModelProfile) -> Agent[AgentDeps, RiskAssessment]:
    agent: Agent[AgentDeps, RiskAssessment] = Agent(
        profile.model,
        output_type=profile.output_type(RiskAssessment),
        instructions=INSTRUCTIONS,
        deps_type=AgentDeps,
        name="risk",
        retries=2,
    )
    if profile.native_tools:

        @agent.tool
        async def search_policy(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search the internal credit policy. Returns the most relevant sections the current role may read."""
            hits = ctx.deps.retriever.search(query, role=ctx.deps.role, k=4)
            return ctx.deps.retriever.format_context(hits) or "no matching policy section"

    return agent
