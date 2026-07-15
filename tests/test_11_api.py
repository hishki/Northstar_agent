"""API-shape tests only -- the orchestrator is replaced with a stub via
FastAPI's dependency-injection override, so these never touch a live model,
Qdrant, or the real document corpus."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app, get_runtime
from app.security.auth import AgentPrincipal, require_agent
from tests.fakes import FakeStructuredStore
from app.schemas import AgentResponse, Citation


class StubRuntime:
    def __init__(self, response: AgentResponse):
        self.structured_store = FakeStructuredStore()
        self._response = response
        self.calls: list[tuple] = []

    def chat(self, message, conversation_id, customer_id=None):
        self.calls.append((message, conversation_id, customer_id))
        return self._response

    def chat_with_trace(self, message, conversation_id, customer_id=None):
        self.calls.append((message, conversation_id, customer_id))
        return self._response, [], {}


def _client(response: AgentResponse) -> tuple[TestClient, StubRuntime]:
    stub = StubRuntime(response)
    app.dependency_overrides[get_runtime] = lambda: stub
    # These tests are about routing/response-shape, not auth -- stub out
    # require_agent the same way get_runtime is stubbed, so they don't need
    # to configure AGENT_API_KEYS or pass an Authorization header.
    app.dependency_overrides[require_agent] = lambda: AgentPrincipal(agent_id="test-agent")
    return TestClient(app), stub


def teardown_function():
    app.dependency_overrides.clear()


def test_chat_matches_sample_api_contract_shape():
    canned = AgentResponse(
        answer="Yes. Cedar Finance has a dedicated technical account manager.",
        citations=[Citation(source="customers.csv", record_id="CUST-1003", excerpt="dedicated_tam=true")],
        grounded=True,
        latency_ms=12.3,
    )
    client, stub = _client(canned)

    resp = client.post(
        "/chat",
        json={"message": "Does Cedar Finance have a dedicated TAM?", "conversation_id": "demo-123", "customer_id": "CUST-1003"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"request", "response"}
    assert body["request"] == {
        "message": "Does Cedar Finance have a dedicated TAM?",
        "conversation_id": "demo-123",
        "customer_id": "CUST-1003",
    }
    assert body["response"]["answer"] == canned.answer
    assert body["response"]["grounded"] is True
    assert body["response"]["citations"][0]["source"] == "customers.csv"
    assert body["response"]["latency_ms"] == 12.3
    # conversation_id reaching the runtime is namespaced by agent_id (here
    # "test-agent", from the require_agent override above) -- see
    # app/api.py's conversation-hijack fix. The client-facing contract
    # (body["request"]["conversation_id"] above) stays un-namespaced.
    assert stub.calls == [("Does Cedar Finance have a dedicated TAM?", "test-agent:demo-123", "CUST-1003")]


def test_chat_customer_id_optional():
    client, stub = _client(AgentResponse(answer="...", citations=[], grounded=False, latency_ms=1.0))

    resp = client.post("/chat", json={"message": "What is the refund window?", "conversation_id": "demo-456"})

    assert resp.status_code == 200
    assert resp.json()["request"]["customer_id"] is None
    assert stub.calls == [("What is the refund window?", "test-agent:demo-456", None)]


def test_sources_endpoint():
    client, _ = _client(AgentResponse(answer="", citations=[], grounded=False, latency_ms=0.0))

    resp = client.get("/sources")

    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()}
    assert names == {"customers.csv", "plans.csv"}


def test_health_endpoint():
    client, _ = _client(AgentResponse(answer="", citations=[], grounded=False, latency_ms=0.0))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
