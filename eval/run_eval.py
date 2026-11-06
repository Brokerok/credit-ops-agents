"""Compare models on the golden set: decision accuracy, extraction accuracy, latency, tokens, cost.

  python -m eval.run_eval --models ollama:llama3.1:8b,anthropic:claude-sonnet-4-5 [--out eval/results.md]

Writes a markdown table you can paste into a decision doc. Each model runs the full pipeline (both agents),
so the numbers reflect what the business would actually see, not a single prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import date
from pathlib import Path

from app.models_registry import build_model
from app.orchestrator import Pipeline
from app.rag.retriever import PolicyRetriever

GOLDEN = Path(__file__).parent / "golden.jsonl"
TODAY = date(2026, 11, 6)


async def run_model(spec: str, cases: list[dict], retriever: PolicyRetriever) -> dict:
    pipe = Pipeline(profile=build_model(spec), retriever=retriever)
    rows = []
    for c in cases:
        res = await pipe.run(c["text"], source=f"eval:{c['id']}", today=TODAY)
        ex = res.extracted
        exp = c["expected"]
        extraction_ok = (
            bool(ex)
            and ex.requested_amount_uah == exp["requested_amount_uah"]
            and ex.vehicle.year == exp["vehicle_year"]
            and ex.vehicle.make.lower() == exp["vehicle_make"]
        )
        rows.append(
            {
                "id": c["id"],
                "expected": c["expected_decision"],
                "got": res.decision.value,
                "decision_ok": res.decision.value == c["expected_decision"],
                "extraction_ok": extraction_ok,
                "latency_ms": res.latency_ms,
                "in": res.input_tokens,
                "out": res.output_tokens,
                "cost": res.cost_usd,
                "errors": res.errors,
                "citations": len(res.risk.citations) if res.risk else 0,
            }
        )
    return {
        "model": spec,
        "rows": rows,
        "decision_acc": sum(r["decision_ok"] for r in rows) / len(rows),
        "extraction_acc": sum(r["extraction_ok"] for r in rows) / len(rows),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
        "tokens": sum(r["in"] + r["out"] for r in rows),
        "cost": sum(r["cost"] for r in rows),
        "cited": sum(r["citations"] > 0 for r in rows) / len(rows),
    }


def to_markdown(results: list[dict], n: int) -> str:
    lines = [
        f"| model | decision acc | extraction acc | runs with citations | p50 latency | tokens | cost ({n} cases) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['decision_acc']:.0%} | {r['extraction_acc']:.0%} | {r['cited']:.0%} | "
            f"{r['p50_latency_ms'] / 1000:.1f}s | {r['tokens']:,} | ${r['cost']:.4f} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"### {r['model']}")
        lines.append("| case | expected | got | extraction | latency | errors |")
        lines.append("|---|---|---|---|---|---|")
        for row in r["rows"]:
            lines.append(
                f"| {row['id']} | {row['expected']} | {row['got']} | {'ok' if row['extraction_ok'] else 'miss'} | "
                f"{row['latency_ms'] / 1000:.1f}s | {'; '.join(row['errors'])[:80]} |"
            )
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model specs")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.md"))
    a = ap.parse_args()
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = PolicyRetriever(engine=None)
    retriever.reindex()
    results = [await run_model(spec.strip(), cases, retriever) for spec in a.models.split(",")]
    md = to_markdown(results, len(cases))
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    asyncio.run(main())
