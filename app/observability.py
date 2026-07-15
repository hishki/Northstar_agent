"""Langfuse tracing for `/chat` requests.

Why this exists: this codebase has zero application observability anywhere
-- there's no `import logging`, no `logger.*` call, no tracing under `app/`.
The closest thing to a trace today is `AgentRuntime.chat_with_trace()`
(`app/agent/graph.py`), which already computes a full tool-call log *and*
token usage on every turn -- but until now `app/api.py`'s `/chat` handler
called the plain `.chat()` wrapper, which silently discards both
(`response, _tool_call_log, _token_usage = self.chat_with_trace(...)`). This
module doesn't add new instrumentation; it ships what `chat_with_trace` was
already computing to Langfuse as one trace per turn, with one child span per
tool call, so the full tool-calling loop is visible as a trace tree instead
of a single flat line.

Deliberately NOT using `langfuse.langchain.CallbackHandler`: that class hard
-imports the full `langchain` package (`from langchain.callbacks.base import
...`), not `langchain-core`, which is the only LangChain package this
project depends on (see `requirements.txt` -- `langchain-core` /
`langchain-ollama` / `langgraph`, no `langchain`). Pulling in all of
`langchain` just for a callback handler would be exactly the dependency
bloat this project avoids elsewhere. Instead this module talks to Langfuse's
native v3 client API directly (`Langfuse().start_as_current_span(...)` +
`update_current_trace(...)`) -- no `langchain` import at all.

Security-relevant signal this closes: `HeuristicSanitizer.is_suspicious()`
(`app/security/sanitizer.py`) already flags likely prompt-injection content
per retrieved chunk, but today that flag only ever reaches the LLM's own
tool-call output -- nothing surfaces it anywhere a security reviewer could
see it. `trace_chat_turn` below attaches it to the trace as an
`injection_flagged` metadata field and as a tag, so it's filterable in the
Langfuse UI.

Graceful degradation: with `config.observability.enabled=False`, or with
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` simply unset (e.g. local dev,
CI, tests), the Langfuse SDK itself detects it has no credentials, logs a
warning, and turns every span/trace call into a no-op rather than raising --
verified directly against langfuse==3.7.0 in this repo's venv. This module
also wraps its call site in a `try/except` as a defensive second layer
(matching this codebase's existing best-effort philosophy, e.g.
`aggregate_token_usage` in `app/agent/graph.py`), so a Langfuse-side hiccup
(bad host, network error, SDK bug) can never take down a `/chat` request.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from langfuse import Langfuse

from app.config import AppConfig
from app.schemas import AgentResponse, ToolCallRecord

# Tools whose `output` can carry a per-chunk `"suspicious"` flag from the
# sanitizer -- see app/agent/tools.py::search_documents/get_document_context.
_RETRIEVAL_TOOL_NAMES = frozenset({"search_documents", "get_document_context"})

_client: Optional[Langfuse] = None
_client_config_id: Optional[int] = None


def get_langfuse_client(config: AppConfig) -> Langfuse:
    """Lazy module-level Langfuse client singleton.

    Reads `public_key` / `secret_key` / `host` from the environment via the
    `*_env` indirection this project uses everywhere else (same convention
    as `AppConfig.llm_base_url()` in `app/config.py`) and builds one
    `Langfuse` instance, reused across requests.

    Deliberately does not hand-roll a separate "disabled" branch: when
    `config.observability.enabled` is False, or the env vars are simply
    unset, we still construct a real `Langfuse(...)` client -- the SDK
    itself detects the missing/absent credentials and no-ops internally
    (see module docstring), which is simpler and less duplicative than a
    second disablement mechanism here. `config.observability.provider` is
    consulted only to decide *whether* to build a Langfuse client at all
    (a future `provider: none` should not import/construct one).
    """
    global _client, _client_config_id
    if config.observability.provider != "langfuse":
        return None  # type: ignore[return-value]

    # Cache keyed on identity of the (typically lru_cached, singleton)
    # config object -- cheap and avoids rebuilding a client per-request.
    if _client is not None and _client_config_id == id(config):
        return _client

    _client = Langfuse(
        public_key=os.environ.get(config.observability.public_key_env),
        secret_key=os.environ.get(config.observability.secret_key_env),
        host=os.environ.get(config.observability.host_env),
        tracing_enabled=config.observability.enabled,
    )
    _client_config_id = id(config)
    return _client


def _injection_flagged(tool_call_log: list[ToolCallRecord]) -> bool:
    """True if any `search_documents`/`get_document_context` tool output in
    this turn contains a chunk the sanitizer flagged `suspicious: true`."""
    for record in tool_call_log:
        if record.tool_name not in _RETRIEVAL_TOOL_NAMES:
            continue
        output = record.output
        # search_documents returns list[dict]; get_document_context returns
        # a single dict -- normalize to a list so both shapes are checked
        # the same way.
        candidates = output if isinstance(output, list) else [output]
        if any(isinstance(item, dict) and item.get("suspicious") for item in candidates):
            return True
    return False


def trace_chat_turn(
    client: Optional[Langfuse],
    *,
    agent_id: str,
    customer_id: Optional[str],
    conversation_id: str,
    question: str,
    tool_call_log: list[ToolCallRecord],
    token_usage: dict[str, int],
    response: AgentResponse,
    latency_ms: float,
) -> None:
    """Emit one Langfuse trace for a completed `/chat` turn: a parent
    `chat_turn` span carrying the question/answer plus one child span per
    tool call, so the tool-calling loop shows up as a tree in the Langfuse
    UI.

    Called from `app/api.py` right after `AgentRuntime.chat_with_trace`
    returns. Best-effort: any exception raised while talking to Langfuse
    (bad host, network error, SDK bug) is swallowed here so a `/chat`
    request can never fail because tracing failed.
    """
    if client is None:
        return
    try:
        injection_flagged = _injection_flagged(tool_call_log)
        with client.start_as_current_span(
            name="chat_turn",
            input={"question": question, "customer_id": customer_id},
            output={"answer": response.answer, "grounded": response.grounded},
        ) as span:
            client.update_current_trace(
                user_id=agent_id,
                session_id=conversation_id,
                metadata={
                    "token_usage": token_usage,
                    "citation_count": len(response.citations),
                    "injection_flagged": injection_flagged,
                    "latency_ms": latency_ms,
                },
                tags=["grounded"] if response.grounded else ["ungrounded"],
            )
            for record in tool_call_log:
                _trace_tool_call(client, record)
        client.flush()
    except Exception:
        # Tracing must never break the request it's observing.
        pass


def _trace_tool_call(client: Langfuse, record: ToolCallRecord) -> Any:
    with client.start_as_current_span(
        name=f"tool:{record.tool_name}",
        input=record.input,
        output=record.output,
    ):
        pass
