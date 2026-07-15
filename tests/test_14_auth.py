"""Auth + audit-tracing tests for the API layer.

Follows tests/test_11_api.py's conventions: FastAPI TestClient +
`app.dependency_overrides` for the seams this codebase already treats as
DI boundaries (`get_runtime`, and now `get_config`/`require_agent`). No live
model, Qdrant, or Langfuse backend is touched -- Langfuse tracing is
exercised by monkeypatching `app.api.trace_chat_turn` (a spy), not by
reaching a real server.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app, get_runtime
from app.config import AppConfig, AuthConfig
from app.schemas import AgentResponse, Citation
from app.security.auth import get_config, require_agent


class StubRuntime:
    """Same shape as tests/test_11_api.py's StubRuntime, but keeps
    `chat_with_trace` (rather than `chat`) since that's the only method
    app/api.py's /chat handler calls now."""

    def __init__(self, response: AgentResponse):
        self._response = response
        self.calls: list[tuple] = []

    def chat_with_trace(self, message, conversation_id, customer_id=None):
        self.calls.append((message, conversation_id, customer_id))
        return self._response, [], {"prompt_eval_count": 10, "eval_count": 5}


_CANNED = AgentResponse(
    answer="Yes, Cedar Finance has a dedicated TAM.",
    citations=[Citation(source="customers.csv", record_id="CUST-1003", excerpt="dedicated_tam=true")],
    grounded=True,
    latency_ms=5.0,
)


def _client(response: AgentResponse = _CANNED) -> tuple[TestClient, StubRuntime]:
    stub = StubRuntime(response)
    app.dependency_overrides[get_runtime] = lambda: stub
    return TestClient(app), stub


def teardown_function():
    app.dependency_overrides.clear()


AUTH_ENABLED_CONFIG = AppConfig(auth=AuthConfig(enabled=True, keys_env="AGENT_API_KEYS"))
AUTH_DISABLED_CONFIG = AppConfig(auth=AuthConfig(enabled=False, keys_env="AGENT_API_KEYS"))


def _use_real_auth(monkeypatch, keys: str, config: AppConfig = AUTH_ENABLED_CONFIG):
    """Exercise the real require_agent dependency (not overridden) against a
    known AGENT_API_KEYS value, via the same dependency_overrides pattern
    already used for get_runtime -- override get_config (require_agent's own
    sub-dependency) rather than mutating global env state for the config
    flag itself; the key *value* still has to live in the env var, since
    that's the contract require_agent implements."""
    monkeypatch.setenv("AGENT_API_KEYS", keys)
    app.dependency_overrides[get_config] = lambda: config


def test_chat_missing_authorization_header_401(monkeypatch):
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice")

    resp = client.post("/chat", json={"message": "hi", "conversation_id": "c1"})

    assert resp.status_code == 401
    assert stub.calls == []


def test_chat_malformed_authorization_header_401(monkeypatch):
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice")

    resp = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "c1"},
        headers={"Authorization": "sk-alice"},  # missing "Bearer " scheme
    )

    assert resp.status_code == 401
    assert stub.calls == []


def test_chat_unknown_key_401(monkeypatch):
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice")

    resp = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "c1"},
        headers={"Authorization": "Bearer sk-not-a-real-key"},
    )

    assert resp.status_code == 401
    assert stub.calls == []


def test_chat_valid_key_succeeds_and_resolves_agent_id(monkeypatch):
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice,sk-bob:agent_bob")
    calls = []
    monkeypatch.setattr(
        "app.api.trace_chat_turn",
        lambda client, **kwargs: calls.append(kwargs),
    )

    resp = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "c1"},
        headers={"Authorization": "Bearer sk-alice"},
    )

    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0]["agent_id"] == "agent_alice"
    # conversation_id reaching the runtime is namespaced by agent_id.
    assert stub.calls[0][1] == "agent_alice:c1"


def test_chat_auth_disabled_requires_no_key(monkeypatch):
    client, stub = _client()
    # No AGENT_API_KEYS set at all, and no Authorization header -- should
    # still succeed because auth.enabled=False fully no-ops require_agent.
    monkeypatch.delenv("AGENT_API_KEYS", raising=False)
    app.dependency_overrides[get_config] = lambda: AUTH_DISABLED_CONFIG

    resp = client.post("/chat", json={"message": "hi", "conversation_id": "c1"})

    assert resp.status_code == 200
    assert stub.calls == [("hi", "anonymous:c1", None)]


def test_health_endpoint_requires_no_auth():
    client, _ = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_conversation_id_namespacing_isolates_agents(monkeypatch):
    """Two different agents supplying the identical client-side
    conversation_id must not share a LangGraph thread -- verified here by
    checking the actual conversation_id/thread_id value that reaches
    AgentRuntime.chat_with_trace for each request."""
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice,sk-bob:agent_bob")

    resp_alice = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "shared-convo"},
        headers={"Authorization": "Bearer sk-alice"},
    )
    resp_bob = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "shared-convo"},
        headers={"Authorization": "Bearer sk-bob"},
    )

    assert resp_alice.status_code == 200
    assert resp_bob.status_code == 200
    # Both clients see the same un-namespaced conversation_id echoed back --
    # the namespacing must not leak into the public API contract.
    assert resp_alice.json()["request"]["conversation_id"] == "shared-convo"
    assert resp_bob.json()["request"]["conversation_id"] == "shared-convo"

    conversation_ids_seen = [call[1] for call in stub.calls]
    assert conversation_ids_seen == ["agent_alice:shared-convo", "agent_bob:shared-convo"]
    assert len(set(conversation_ids_seen)) == 2


def test_chat_tracing_helper_invoked_with_expected_data(monkeypatch):
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice")
    calls = []
    monkeypatch.setattr(
        "app.api.trace_chat_turn",
        lambda client, **kwargs: calls.append(kwargs),
    )

    resp = client.post(
        "/chat",
        json={"message": "Does Cedar Finance have a dedicated TAM?", "conversation_id": "c1", "customer_id": "CUST-1003"},
        headers={"Authorization": "Bearer sk-alice"},
    )

    assert resp.status_code == 200
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["agent_id"] == "agent_alice"
    assert kwargs["customer_id"] == "CUST-1003"
    # trace_chat_turn receives the original, un-namespaced conversation_id --
    # namespacing is purely a runtime/checkpointer concern, not a tracing one.
    assert kwargs["conversation_id"] == "c1"
    assert kwargs["question"] == "Does Cedar Finance have a dedicated TAM?"
    assert kwargs["response"].grounded is True
    assert kwargs["token_usage"] == {"prompt_eval_count": 10, "eval_count": 5}


def test_chat_succeeds_with_langfuse_misconfigured(monkeypatch):
    """Graceful degradation: no LANGFUSE_PUBLIC_KEY/SECRET_KEY set at all --
    /chat must still return 200. The Langfuse SDK itself no-ops tracing
    when credentials are absent (verified directly against the installed
    langfuse package); trace_chat_turn also wraps its own logic in
    try/except as a defensive second layer."""
    client, stub = _client()
    _use_real_auth(monkeypatch, "sk-alice:agent_alice")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    resp = client.post(
        "/chat",
        json={"message": "hi", "conversation_id": "c1"},
        headers={"Authorization": "Bearer sk-alice"},
    )

    assert resp.status_code == 200
    assert len(stub.calls) == 1
