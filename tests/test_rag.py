from pathlib import Path

from app.rag.chunker import load_policy_dir
from app.rag.retriever import PolicyRetriever

POLICY = Path("app/policy")


def test_chunker_produces_section_level_chunks_with_roles():
    chunks = load_policy_dir(POLICY)
    ids = {c.section.split()[0] for c in chunks}
    assert {"CP-3.2", "CP-4", "CR-1", "SP-3"} <= ids
    cp = next(c for c in chunks if c.section.startswith("CP-3.2"))
    assert cp.roles == ("risk", "legal")
    assert "60%" in cp.content


def test_retrieval_is_role_aware():
    r = PolicyRetriever(engine=None, policy_dir=str(POLICY))
    assert r.backend == "memory"
    assert r.reindex() > 5
    risk_hits = r.search("maximum loan to value LTV limit", role="risk", k=3)
    assert risk_hits and any(h.chunk.section.startswith("CP-3.2") for h in risk_hits)
    # the sales playbook is invisible to the risk role, and credit policy is invisible to sales
    assert all(h.chunk.doc != "sales_playbook" for h in r.search("discount authority", role="risk", k=5))
    sales_hits = r.search("discount authority for repeat clients", role="sales", k=5)
    assert sales_hits and all(h.chunk.doc != "credit_policy" for h in sales_hits)
    assert r.search("anything", role="nobody") == []


def test_context_format_carries_citable_ids():
    r = PolicyRetriever(engine=None, policy_dir=str(POLICY))
    r.reindex()
    ctx = r.format_context(r.search("vehicle age 15 years", role="risk", k=2))
    assert "[credit_policy | CP-4" in ctx
