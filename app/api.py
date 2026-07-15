"""HTTP API. Shape matches `sample_api_contract.json` exactly: POST /chat
takes {message, conversation_id, customer_id} and returns
{request: {...}, response: {answer, citations, grounded, latency_ms}}.

`get_runtime` is the FastAPI dependency-injection seam: production takes the
real cached `AgentRuntime` (which loads the embedding model, indexes the
corpus, and constructs the Ollama chat model on first call), tests override
it via `app.dependency_overrides[get_runtime]` with a stub -- no live model
or Qdrant needed to test routing/response-shape.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterator, Optional

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import AgentRuntime
from app.config import AppConfig
from app.observability import get_langfuse_client, trace_chat_turn
from app.schemas import AgentResponse, SourceInfo
from app.security.auth import AgentPrincipal, get_config, require_agent

app = FastAPI(title="Northstar Cloud Support Agent")


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    customer_id: Optional[str] = None


class ChatEnvelope(BaseModel):
    request: ChatRequest
    response: AgentResponse


@lru_cache(maxsize=1)
def get_runtime() -> AgentRuntime:
    """Built lazily on first request (not at import time) so merely
    importing this module -- e.g. for OpenAPI schema generation -- doesn't
    require the embedding model or a reachable Ollama/Qdrant."""
    return AgentRuntime(get_config())


@app.post("/chat", response_model=ChatEnvelope)
def chat(
    payload: ChatRequest,
    runtime: AgentRuntime = Depends(get_runtime),
    config: AppConfig = Depends(get_config),
    principal: AgentPrincipal = Depends(require_agent),
) -> ChatEnvelope:
    # Namespace the conversation_id handed to the runtime/checkpointer by
    # agent_id so two different agents supplying the same client-side
    # conversation_id string don't share a LangGraph thread (a
    # conversation-hijack risk once /chat is multi-agent). The client only
    # ever sees its own un-namespaced value -- ChatEnvelope.request below
    # echoes back `payload` as-is, so this never leaks into the public API
    # contract.
    namespaced_conversation_id = f"{principal.agent_id}:{payload.conversation_id}"
    response, tool_call_log, token_usage = runtime.chat_with_trace(
        message=payload.message,
        conversation_id=namespaced_conversation_id,
        customer_id=payload.customer_id,
    )
    langfuse_client = get_langfuse_client(config)
    trace_chat_turn(
        langfuse_client,
        agent_id=principal.agent_id,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        question=payload.message,
        tool_call_log=tool_call_log,
        token_usage=token_usage,
        response=response,
        latency_ms=response.latency_ms,
    )
    return ChatEnvelope(request=payload, response=response)


@app.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    runtime: AgentRuntime = Depends(get_runtime),
    config: AppConfig = Depends(get_config),
    principal: AgentPrincipal = Depends(require_agent),
) -> StreamingResponse:
    """Server-Sent Events variant of `/chat`. Phase 1 only: this streams
    *progress* events (which node/tool the agent<->tools loop just ran)
    while the turn is in flight, killing the blank-spinner wait -- it does
    NOT stream the final answer text token-by-token. That's a harder
    follow-up: the model's answer arrives as the `submit_answer` tool
    call's `answer` argument, streamed (if at all) as partial tool-call-args
    JSON rather than plain chat tokens, which needs its own
    JSON-reconstruction handling and is out of scope here.

    Behind the same `require_agent` dependency as `/chat` (not an
    unauthenticated backdoor around it), and applies the same
    agent-namespaced `conversation_id` before handing it to the runtime.
    Hand-rolls the `text/event-stream` wire format (`event: <type>\\ndata:
    <json>\\n\\n` per event -- the standard SSE shape, matching what
    `sse-starlette` itself emits) rather than adding a new dependency for a
    few lines of string formatting, consistent with this project's minimal-
    dependency approach elsewhere.

    Once `AgentRuntime.chat_stream` yields its terminal `"done"` event, this
    calls `trace_chat_turn` with that event's `tool_call_log`/`token_usage`
    -- streaming gets the same Langfuse trace treatment `/chat` does, not a
    silent skip.
    """
    namespaced_conversation_id = f"{principal.agent_id}:{payload.conversation_id}"

    def event_source() -> Iterator[str]:
        for event in runtime.chat_stream(
            message=payload.message,
            conversation_id=namespaced_conversation_id,
            customer_id=payload.customer_id,
        ):
            if event["type"] == "done":
                response = AgentResponse(**event["response"])
                langfuse_client = get_langfuse_client(config)
                trace_chat_turn(
                    langfuse_client,
                    agent_id=principal.agent_id,
                    customer_id=payload.customer_id,
                    conversation_id=payload.conversation_id,
                    question=payload.message,
                    tool_call_log=event["tool_call_log"],
                    token_usage=event["token_usage"],
                    response=response,
                    latency_ms=response.latency_ms,
                )
                payload_json = json.dumps({"type": "done", "response": event["response"]})
                yield f"event: done\ndata: {payload_json}\n\n"
            else:
                yield f"event: step\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/sources", response_model=list[SourceInfo])
def sources(
    runtime: AgentRuntime = Depends(get_runtime),
    principal: AgentPrincipal = Depends(require_agent),
) -> list[SourceInfo]:
    return runtime.structured_store.list_sources()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
