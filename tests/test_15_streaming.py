"""Tests for `POST /chat/stream`: SSE progress events + a final `done` event
carrying the same `AgentResponse` shape non-streaming `/chat` returns.

Follows tests/test_11_api.py's/test_14_auth.py's convention: FastAPI
TestClient + `app.dependency_overrides` for `get_runtime`/`get_config`/
`require_agent` -- no live model, Qdrant, or Langfuse backend touched.
"""
from __future__ import annotations

import json
from datetime import date

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import app, get_runtime
from app.config import AppConfig, AuthConfig
from app.schemas import AgentResponse, Citation
from app.security.auth import AgentPrincipal, get_config, require_agent
from tests.fakes import FakeSanitizer, FakeStructuredStore


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse a raw `text/event-stream` body back into `(event_type, data)`
    pairs, one per `event: ...\\ndata: ...\\n\\n` block."""
    events = []
    for block in text.strip("\n").split("\n\n"):
        if not block.strip():
            continue
        event_type = None
        data_line = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data_line = line[len("data: ") :]
        assert event_type is not None and data_line is not None, f"malformed SSE block: {block!r}"
        events.append((event_type, json.loads(data_line)))
    return events


_CANNED = AgentResponse(
    answer="The monthly refund window is 7 days.",
    citations=[Citation(source="fake_doc.md", section="Refund Policy", excerpt="Monthly refunds within 7 days.")],
    grounded=True,
    latency_ms=12.3,
)


class StubRuntime:
    """Same shape as tests/test_11_api.py's StubRuntime, plus a `chat_stream`
    generator driving `/chat/stream` without a live LLM or compiled
    LangGraph graph at all -- a plain canned event sequence is enough to
    exercise the API route's SSE formatting/auth/tracing wiring."""

    def __init__(self, response: AgentResponse):
        self.structured_store = FakeStructuredStore()
        self._response = response
        self.calls: list[tuple] = []
        self.stream_calls: list[tuple] = []

    def chat_with_trace(self, message, conversation_id, customer_id=None):
        self.calls.append((message, conversation_id, customer_id))
        return self._response, [], {}

    def chat_stream(self, message, conversation_id, customer_id=None):
        self.stream_calls.append((message, conversation_id, customer_id))
        yield {"type": "step", "node": "agent", "detail": "thinking"}
        yield {"type": "step", "node": "tools", "detail": "called search_documents"}
        yield {"type": "step", "node": "agent", "detail": "thinking"}
        yield {"type": "step", "node": "validate_citations", "detail": "verifying citations"}
        yield {
            "type": "done",
            "response": self._response.model_dump(),
            "tool_call_log": [],
            "token_usage": {"prompt_eval_count": 10, "eval_count": 5},
        }


def _client(response: AgentResponse = _CANNED) -> tuple[TestClient, StubRuntime]:
    stub = StubRuntime(response)
    app.dependency_overrides[get_runtime] = lambda: stub
    app.dependency_overrides[require_agent] = lambda: AgentPrincipal(agent_id="test-agent")
    return TestClient(app), stub


def teardown_function():
    app.dependency_overrides.clear()


AUTH_ENABLED_CONFIG = AppConfig(auth=AuthConfig(enabled=True, keys_env="AGENT_API_KEYS"))


def test_chat_stream_event_shape_is_valid_sse():
    client, stub = _client()

    resp = client.post(
        "/chat/stream",
        json={"message": "What is the refund window?", "conversation_id": "demo-123"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    assert len(events) == 5
    # Every event but the last is a "step"; exactly one trailing "done".
    assert [etype for etype, _ in events[:-1]] == ["step"] * 4
    assert events[-1][0] == "done"
    for etype, data in events[:-1]:
        assert data["type"] == "step"
        assert "node" in data
        assert "detail" in data
    assert stub.stream_calls == [("What is the refund window?", "test-agent:demo-123", None)]


def test_chat_stream_done_event_matches_non_streaming_response_shape():
    client, stub = _client()

    stream_resp = client.post(
        "/chat/stream",
        json={"message": "What is the refund window?", "conversation_id": "demo-123"},
    )
    chat_resp = client.post(
        "/chat",
        json={"message": "What is the refund window?", "conversation_id": "demo-123"},
    )

    events = _parse_sse(stream_resp.text)
    done_response = events[-1][1]["response"]
    chat_response = chat_resp.json()["response"]

    assert set(done_response.keys()) == set(chat_response.keys())
    assert done_response["answer"] == chat_response["answer"] == _CANNED.answer
    assert done_response["grounded"] == chat_response["grounded"] is True
    assert done_response["citations"] == chat_response["citations"]
    assert done_response["latency_ms"] == chat_response["latency_ms"] == _CANNED.latency_ms


def test_chat_stream_tool_call_sequence_reflected_in_step_details():
    client, _ = _client()

    resp = client.post(
        "/chat/stream",
        json={"message": "What is the refund window?", "conversation_id": "demo-123"},
    )

    events = _parse_sse(resp.text)
    tool_steps = [data for etype, data in events if etype == "step" and data["node"] == "tools"]
    assert len(tool_steps) == 1
    assert tool_steps[0]["detail"] == "called search_documents"


def test_chat_stream_missing_authorization_header_401(monkeypatch):
    stub = StubRuntime(_CANNED)
    app.dependency_overrides[get_runtime] = lambda: stub
    monkeypatch.setenv("AGENT_API_KEYS", "sk-alice:agent_alice")
    app.dependency_overrides[get_config] = lambda: AUTH_ENABLED_CONFIG
    client = TestClient(app)

    resp = client.post("/chat/stream", json={"message": "hi", "conversation_id": "c1"})

    assert resp.status_code == 401
    assert stub.stream_calls == []


def test_chat_stream_bad_key_401(monkeypatch):
    stub = StubRuntime(_CANNED)
    app.dependency_overrides[get_runtime] = lambda: stub
    monkeypatch.setenv("AGENT_API_KEYS", "sk-alice:agent_alice")
    app.dependency_overrides[get_config] = lambda: AUTH_ENABLED_CONFIG
    client = TestClient(app)

    resp = client.post(
        "/chat/stream",
        json={"message": "hi", "conversation_id": "c1"},
        headers={"Authorization": "Bearer sk-not-a-real-key"},
    )

    assert resp.status_code == 401
    assert stub.stream_calls == []


def test_chat_stream_valid_key_succeeds_and_namespaces_conversation_id(monkeypatch):
    stub = StubRuntime(_CANNED)
    app.dependency_overrides[get_runtime] = lambda: stub
    monkeypatch.setenv("AGENT_API_KEYS", "sk-alice:agent_alice")
    app.dependency_overrides[get_config] = lambda: AUTH_ENABLED_CONFIG
    client = TestClient(app)

    resp = client.post(
        "/chat/stream",
        json={"message": "hi", "conversation_id": "c1"},
        headers={"Authorization": "Bearer sk-alice"},
    )

    assert resp.status_code == 200
    assert stub.stream_calls == [("hi", "agent_alice:c1", None)]


def test_chat_stream_calls_trace_chat_turn_on_done(monkeypatch):
    stub = StubRuntime(_CANNED)
    app.dependency_overrides[get_runtime] = lambda: stub
    app.dependency_overrides[require_agent] = lambda: AgentPrincipal(agent_id="test-agent")
    calls = []
    monkeypatch.setattr("app.api.trace_chat_turn", lambda client, **kwargs: calls.append(kwargs))
    client = TestClient(app)

    resp = client.post(
        "/chat/stream",
        json={"message": "What is the refund window?", "conversation_id": "c1"},
    )

    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0]["agent_id"] == "test-agent"
    assert calls[0]["conversation_id"] == "c1"  # un-namespaced, same as /chat
    assert calls[0]["question"] == "What is the refund window?"
    assert calls[0]["response"].grounded is True
    assert calls[0]["token_usage"] == {"prompt_eval_count": 10, "eval_count": 5}


# ---------------------------------------------------------------------------
# Deeper test: drive the *real* AgentRuntime.chat_stream generator (real
# build_agent_graph, real validate_citations/verification logic) against a
# scripted fake chat model, the same pattern tests/test_10_orchestrator_unit.py
# uses to exercise the graph without a live Ollama server.
# ---------------------------------------------------------------------------


class _FakeChatModel:
    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)

    def bind_tools(self, tools):  # noqa: ARG002 - fake ignores tool schemas
        return self

    def invoke(self, messages):
        return self._responses.pop(0)

    def with_structured_output(self, schema, method="json_schema", **kwargs):  # noqa: ARG002
        raise NotImplementedError("not exercised by this test")


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _submit(answer, citations=None, call_id="submit_1"):
    return _tool_call("submit_answer", {"answer": answer, "citations": citations or []}, call_id)


def test_agent_runtime_chat_stream_real_graph_yields_steps_then_done():
    from unittest.mock import MagicMock, patch

    from app.agent.graph import AgentRuntime
    from app.config import load_config
    from app.retrieval import create_retriever
    from app.schemas import DocChunk

    config = load_config()
    retriever = create_retriever(config)  # unindexed -- AgentRuntime.__init__ indexes it
    structured_store = FakeStructuredStore()
    sanitizer = FakeSanitizer()

    search = _tool_call("search_documents", {"query": "refund window"}, "call_1")
    final = _submit(
        "The monthly refund window is 7 days.",
        [{"source": "fake_doc.md", "section": "Refund Policy", "excerpt": "Monthly refunds within 7 days."}],
    )
    fake_llm = _FakeChatModel([search, final])

    mock_document_store = MagicMock()
    mock_document_store.load_chunks.return_value = [
        DocChunk(
            chunk_id="fake_doc.md#refund_policy",
            source="fake_doc.md",
            section="Refund Policy",
            text="Monthly refunds within 7 days.",
            effective_date=date(2026, 2, 1),
        )
    ]

    with patch("app.agent.graph.build_retriever", return_value=retriever), patch(
        "app.agent.graph.build_document_store", return_value=mock_document_store
    ), patch("app.agent.graph.build_structured_store", return_value=structured_store), patch(
        "app.agent.graph.build_sanitizer", return_value=sanitizer
    ):
        runtime = AgentRuntime(config, llm=fake_llm)

    events = list(runtime.chat_stream("What is the refund window?", conversation_id="stream-1"))

    assert [e["type"] for e in events[:-1]] == ["step"] * (len(events) - 1)
    assert events[-1]["type"] == "done"

    node_sequence = [e["node"] for e in events if e["type"] == "step"]
    # agent -> tools (search_documents) -> agent -> tools (submit_answer) ->
    # validate_citations -- matches build_agent_graph's shape for a
    # search-then-submit turn with no remind/force_submit escalation.
    assert node_sequence == ["agent", "tools", "agent", "tools", "validate_citations"]

    tools_steps = [e for e in events if e["type"] == "step" and e["node"] == "tools"]
    assert tools_steps[0]["detail"] == "called search_documents"
    assert tools_steps[1]["detail"] == "called submit_answer"

    done = events[-1]
    assert done["response"]["answer"] == "The monthly refund window is 7 days."
    assert done["response"]["grounded"] is True
    assert len(done["response"]["citations"]) == 1
    assert len(done["tool_call_log"]) == 2
    assert done["tool_call_log"][0].tool_name == "search_documents"
    assert done["tool_call_log"][1].tool_name == "submit_answer"
    assert "prompt_eval_count" in done["token_usage"]
