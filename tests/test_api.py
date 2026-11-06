import pytest
from httpx import ASGITransport, AsyncClient
from pydantic_ai.models.test import TestModel

from app.api import app
from app.orchestrator import get_pipeline
from tests.test_pipeline import EXTRACTED, RISK


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health_and_submit_and_fetch_run(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json()["model"] == "test:test"
    p = get_pipeline()
    with (
        p.intake.override(model=TestModel(custom_output_args=EXTRACTED)),
        p.risk.override(model=TestModel(custom_output_args=RISK)),
    ):
        r = await client.post("/applications", json={"text": "Olena wants 150000 UAH against a Toyota Camry 2019"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in {"auto_approved", "human_review"}
    r = await client.get(f"/runs/{body['run_id']}")
    assert r.status_code == 200 and r.json()["status"] == "done"
    r = await client.get("/runs")
    assert r.status_code == 200 and len(r.json()) >= 1


async def test_rejects_too_short_text(client: AsyncClient):
    r = await client.post("/applications", json={"text": "hi"})
    assert r.status_code == 422
