"""The LangGraph orchestrator: agent <-> tools loop + a citation-validation
node, wired around whichever chat model / retriever / structured store /
sanitizer the factory builds for the current config.

Graph shape:

    START -> agent --(tool_calls, under loop limit)---> tools --(submit_answer called)--> validate_citations -> END
               |
               |--(no tool_calls, first miss)------> remind -> agent (one free-text nudge)
               |
               |--(no tool_calls, nudge already used)--> force_submit ----------------> validate_citations -> END
               |
               '--(loop limit hit)---------------------------------------------------> validate_citations -> END

`tools` is a hand-rolled node (not LangGraph's prebuilt ToolNode) so it can
also append a `ToolCallRecord` per call to `tool_call_log` -- that log is
what `validate_citations` cross-checks every citation against, so a citation
naming a source the conversation never actually retrieved gets dropped
rather than trusted at face value.

`remind`/`force_submit` handle a failure mode observed live: qwen2.5:7b-
instruct will sometimes write a full (even correct) prose answer as a plain
chat message instead of calling `submit_answer` as instructed. The obvious
fix -- force the tool call via `tool_choice` -- isn't available: Ollama
doesn't support it (`ChatOllama.bind_tools()` documents `tool_choice` as
silently ignored). So the first miss gets a cheap text nudge (`remind`); if
that *also* misses (observed live: the model can go completely blank on a
second nudge rather than comply), `force_submit` escalates to
`with_structured_output(..., method="json_schema")` -- a constraint Ollama
does enforce at the grammar level, so the model cannot produce anything
except valid `{answer, citations}` JSON. This is the actual fix, not another
nudge; the raw-text fallback still exists as a last-resort safety net for
the (unobserved so far) case where even the structured call fails.

The model concludes every turn by calling the `submit_answer` tool (defined
in `app/agent/tools.py`) rather than writing a free-text "Citations:" block.
An earlier version asked for that block and regex-parsed it; running it
against the live qwen2.5:7b-instruct model surfaced two real failure modes
-- some answers had zero citation lines at all, others invented a different
format -- even when the model had retrieved and used the right evidence.
Citations now ride the same tool-calling channel the model already uses
reliably for search_documents/query_plan_data/etc. (which had 100%
well-formed args in that same run), validated by `CitationInput`'s Pydantic
schema before it ever reaches `validate_citations_node`.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.llm import get_chat_model
from app.agent.state import AgentState
from app.agent.system_prompt import SYSTEM_PROMPT
from app.agent.tools import build_tools
from app.config import AppConfig
from app.factory import build_document_store, build_retriever, build_sanitizer, build_structured_store
from app.interfaces import Retriever, Sanitizer, StructuredStore
from app.schemas import AgentResponse, Citation, ToolCallRecord

MAX_AGENT_LOOPS = 6


def _collect_known_sources(tool_call_log: list[ToolCallRecord]) -> dict[str, list[dict]]:
    """source filename -> list of tool-output entries this conversation has
    actually retrieved for it, used to verify (not just format-check)
    citations."""
    known: dict[str, list[dict]] = {}

    def _add(source: Any, entry: dict) -> None:
        if isinstance(source, str) and source:
            known.setdefault(source, []).append(entry)

    for record in tool_call_log:
        output = record.output
        if record.tool_name == "search_documents" and isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    _add(item.get("source"), item)
        elif record.tool_name == "get_document_context" and isinstance(output, dict):
            _add(output.get("source"), output)
        elif record.tool_name == "query_plan_data" and isinstance(output, dict) and "error" not in output:
            fields = output.get("fields", {}) or {}
            # Accept both a JSON dump and the "key=value" shorthand the
            # assignment's own sample_api_contract.json uses for structured
            # citations (e.g. "dedicated_tam=true") -- json.dumps per-value
            # (not str()) so booleans/None render as true/false/null to
            # match that convention exactly, not Python's True/False/None.
            kv_lines = "\n".join(f"{k}={json.dumps(v, default=str)}" for k, v in fields.items())
            entry = {"source": output.get("source"), "content": json.dumps(fields, default=str) + "\n" + kv_lines}
            _add("customers.csv", entry)
            _add("plans.csv", entry)
        elif record.tool_name == "list_sources" and isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    _add(item.get("name"), {"source": item.get("name"), "content": item.get("description") or ""})
    return known


def _extract_submitted_answer(tool_call_log: list[ToolCallRecord]) -> tuple[str, list[Citation]]:
    """Pull the answer + citations out of the most recent successful
    `submit_answer` call. An earlier version had the model write a
    free-text "Citations:" block and regex-parsed it -- running that
    against the live qwen2.5:7b-instruct model surfaced two real failure
    modes (some answers had zero citation lines at all, others invented a
    different format), even when the model had retrieved and used the
    right evidence. Citations now arrive as `CitationInput`-validated tool
    arguments instead (see `app/agent/tools.py`), which the model's
    tool-calling was 100% reliable at in the same live run.

    Falls back to whatever text the last message contains, with no
    citations, if `submit_answer` was never successfully called (e.g. the
    loop-limit guard forced termination, or every attempt failed schema
    validation) -- better to surface *something* than to return an empty
    answer.
    """
    for record in reversed(tool_call_log):
        if record.tool_name != "submit_answer":
            continue
        output = record.output
        if not isinstance(output, dict) or "error" in output:
            continue
        answer = str(output.get("answer", "")).strip()
        citations = [
            Citation(source=c["source"], section=c.get("section"), excerpt=c["excerpt"])
            for c in (output.get("citations") or [])
            if isinstance(c, dict) and c.get("source") and c.get("excerpt")
        ]
        return answer, citations
    return "", []


_UNTRUSTED_TAG_RE = re.compile(r"</?untrusted_document_content[^>]*>")

# Observed live, even after the `remind` nudge: the model sometimes writes
# out what *looks like* the submit_answer call as plain text --
# `submit_answer("...", [...])` -- instead of actually invoking the tool.
# It understood the instruction but expressed it as pseudo-code rather than
# a real structured tool call. Extract just the first quoted string
# argument (the answer) so the user sees a real sentence instead of raw
# pseudo-code; deliberately does NOT attempt to parse the citations argument
# out of this malformed text -- if the model can't manage a real tool call,
# its inline "citations" aren't trustworthy either, so this path always
# yields zero citations (grounded=False), same as any other fallback.
_FAKE_SUBMIT_CALL_RE = re.compile(r'submit_answer\(\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _clean_answer_text(text: str) -> str:
    fake_call = _FAKE_SUBMIT_CALL_RE.search(text)
    if fake_call:
        text = fake_call.group(1).replace('\\"', '"').replace("\\n", "\n")
    return _UNTRUSTED_TAG_RE.sub("", text).strip()


def _fallback_answer_text(state: AgentState) -> str:
    """Used when `submit_answer` was never successfully called this turn
    (loop-limit guard, or the model just answered as plain chat text
    despite rule 2). Observed live: a model that skips `submit_answer` will
    sometimes paste a raw tool result -- <untrusted_document_content> tags
    included -- straight into its chat reply; strip the tag markers so that
    never reaches the user verbatim.

    Searches *backward* through the transcript for the most recent
    non-empty AIMessage rather than only checking the last message.
    Observed live: after the `remind` nudge, the model can go completely
    blank (empty content, no tool call) on every subsequent attempt --
    reading only `state["messages"][-1]` in that case throws away a
    perfectly good answer the model already gave a few turns earlier,
    surfacing nothing instead."""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, AIMessage):
            continue
        cleaned = _clean_answer_text(str(msg.content))
        if cleaned:
            return cleaned
    return ""


def _verify_citations(citations: list[Citation], known_sources: dict[str, list[dict]]) -> list[Citation]:
    """Keep only citations whose source was actually retrieved, and whose
    excerpt is a (case-insensitive) substring of that source's retrieved
    content when we have retrieved content to check against -- this is the
    hallucinated-citation guard."""
    verified = []
    for citation in citations:
        entries = known_sources.get(citation.source)
        if not entries:
            continue
        contents = [str(e.get("content", "")) for e in entries]
        if any(contents) and not any(citation.excerpt.lower() in c.lower() for c in contents if c):
            continue
        verified.append(citation)
    return verified


def _build_human_message(message: str, customer_id: Optional[str]) -> HumanMessage:
    """Prepend the request's `customer_id` (when given) into the visible
    turn so the model can actually act on it.

    `customer_id` was previously only written into `AgentState` and never
    included in what the model reads -- a real bug, found live: asked
    "Does Cedar Finance have a dedicated TAM?" with customer_id=CUST-1003
    set, the model had no way to know that ID, never called
    `query_plan_data`, and guessed (incorrectly) from the general policy
    document alone instead. Support agents are expected to ask about
    customers by name, not by ID, so this note is what makes
    `query_plan_data(customer_id=...)` actually reachable for those
    questions."""
    if not customer_id:
        return HumanMessage(content=message)
    return HumanMessage(
        content=f"[Context: this request is associated with customer_id={customer_id}]\n\n{message}"
    )


_REMIND_MESSAGE = (
    "You did not call the submit_answer tool. You must call submit_answer now, with your answer "
    "and citations based on what you have found so far, to finish this turn."
)


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if state.get("loop_count", 0) >= MAX_AGENT_LOOPS:
        return "validate_citations"
    if tool_calls:
        return "tools"
    # Model answered as plain chat text, ignoring rule 2 (finish by calling
    # submit_answer). One cheap text nudge first; if that also misses,
    # escalate to a call Ollama actually enforces rather than nudging
    # again (tool_choice isn't supported, so a second nudge is just as
    # ignorable as the first -- observed live, the model can go completely
    # blank on it instead of complying).
    if state.get("remind_count", 0) >= 1:
        return "force_submit"
    return "remind"


def _route_after_tools(state: AgentState) -> str:
    return "validate_citations" if state.get("submitted", False) else "agent"


def build_agent_graph(
    config: AppConfig,
    retriever: Retriever,
    structured_store: StructuredStore,
    sanitizer: Sanitizer,
    llm: Optional[BaseChatModel] = None,
) -> CompiledStateGraph:
    """`llm` is an injection seam for tests -- pass a fake chat model to
    drive the graph without a live Ollama server. Production callers
    (`AgentRuntime` below) leave it as None and get `get_chat_model(config)`."""
    chat_model = llm if llm is not None else get_chat_model(config)
    tools = build_tools(retriever, structured_store, sanitizer)
    llm_with_tools = chat_model.bind_tools(list(tools.values()))

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "loop_count": state.get("loop_count", 0) + 1}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages = []
        records = []
        submitted = False
        for call in getattr(last, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {})
            tool_obj = tools.get(name)
            if tool_obj is None:
                result: Any = {"error": f"Unknown tool: {name!r}"}
            else:
                try:
                    result = tool_obj.invoke(args)
                except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model, don't crash the graph
                    result = {"error": str(exc)}
            tool_messages.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"], name=name)
            )
            records.append(ToolCallRecord(tool_name=name, input=args, output=result))
            if name == "submit_answer" and isinstance(result, dict) and "error" not in result:
                submitted = True
        return {"messages": tool_messages, "tool_call_log": records, "submitted": submitted}

    def remind_node(state: AgentState) -> dict:
        return {"messages": [HumanMessage(content=_REMIND_MESSAGE)], "remind_count": state.get("remind_count", 0) + 1}

    def force_submit_node(state: AgentState) -> dict:
        """The actual fix, not another nudge: Ollama doesn't support forcing
        a specific tool call, but it does enforce `with_structured_output`'s
        JSON-schema constraint at the grammar level. Reuses the real
        `submit_answer` tool's own auto-generated arg schema so there is
        exactly one definition of "what a valid answer looks like" -- not a
        second, hand-maintained copy that could drift from it."""
        structured_llm = chat_model.with_structured_output(tools["submit_answer"].args_schema, method="json_schema")
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        try:
            result = structured_llm.invoke(messages)
        except Exception:  # noqa: BLE001 - even the enforced call can fail transport-side; fall back, don't crash
            return {"submitted": False}
        answer = getattr(result, "answer", None)
        raw_citations = getattr(result, "citations", None)
        if answer is None and isinstance(result, dict):
            answer = result.get("answer")
            raw_citations = result.get("citations")
        citations = [c if isinstance(c, dict) else c.model_dump() for c in (raw_citations or [])]
        output = {"answer": answer or "", "citations": citations}
        record = ToolCallRecord(tool_name="submit_answer", input={}, output=output)
        return {"tool_call_log": [record], "submitted": True}

    def validate_citations_node(state: AgentState) -> dict:
        tool_call_log = state.get("tool_call_log", [])
        answer_text, citations = _extract_submitted_answer(tool_call_log)
        if not answer_text and not citations:
            # submit_answer was never successfully called (loop-limit guard,
            # or the model answered as plain text) -- fall back to raw
            # content rather than returning nothing.
            answer_text = _fallback_answer_text(state)
        known_sources = _collect_known_sources(tool_call_log)
        verified = _verify_citations(citations, known_sources)
        return {
            "final_answer": answer_text,
            "final_citations": verified,
            "grounded": len(verified) > 0,
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("remind", remind_node)
    graph.add_node("force_submit", force_submit_node)
    graph.add_node("validate_citations", validate_citations_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "remind": "remind", "force_submit": "force_submit", "validate_citations": "validate_citations"},
    )
    graph.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", "validate_citations": "validate_citations"})
    graph.add_edge("remind", "agent")
    graph.add_edge("force_submit", "validate_citations")
    graph.add_edge("validate_citations", END)
    return graph.compile(checkpointer=MemorySaver())


class AgentRuntime:
    """Owns the compiled graph plus the indexed retriever, and is the single
    entry point the API layer calls."""

    def __init__(self, config: AppConfig, llm: Optional[BaseChatModel] = None):
        self.config = config
        self.retriever = build_retriever(config)
        self.retriever.index(build_document_store(config).load_chunks())
        self.structured_store = build_structured_store(config)
        self.sanitizer = build_sanitizer(config)
        self._graph = build_agent_graph(config, self.retriever, self.structured_store, self.sanitizer, llm=llm)

    def chat(self, message: str, conversation_id: str, customer_id: Optional[str] = None) -> AgentResponse:
        response, _tool_call_log, _token_usage = self.chat_with_trace(message, conversation_id, customer_id)
        return response

    def chat_with_trace(
        self, message: str, conversation_id: str, customer_id: Optional[str] = None
    ) -> tuple[AgentResponse, list[ToolCallRecord], dict[str, int]]:
        """Same as `chat`, but also returns this turn's tool-call log (for a
        true retrieval-recall check, independent of which citations survived
        the hallucination guard) and a best-effort token-usage tally read off
        every AIMessage's `usage_metadata` -- used by `evals/run_eval.py`.
        Not part of the public API contract; `app/api.py` only calls `chat`.
        """
        start = time.monotonic()
        result = self._graph.invoke(
            {
                "messages": [_build_human_message(message, customer_id)],
                "tool_call_log": [],
                "loop_count": 0,
                "submitted": False,
                "remind_count": 0,
                "customer_id": customer_id,
            },
            config={"configurable": {"thread_id": conversation_id}},
        )
        latency_ms = (time.monotonic() - start) * 1000
        response = AgentResponse(
            answer=result.get("final_answer") or "",
            citations=result.get("final_citations") or [],
            grounded=bool(result.get("grounded")),
            latency_ms=latency_ms,
        )
        token_usage = aggregate_token_usage(result.get("messages", []))
        return response, list(result.get("tool_call_log", [])), token_usage


def aggregate_token_usage(messages: list) -> dict[str, int]:
    """Sum `usage_metadata` (input_tokens/output_tokens) across every message
    that carries it, plus Ollama's own generation-timing fields when present
    (`response_metadata.eval_count`/`eval_duration`/`prompt_eval_count`/
    `prompt_eval_duration`, all nanoseconds for the *_duration fields) --
    this is real per-call generation/prefill speed straight from the
    inference engine, not a coarse estimate from wall-clock latency (which
    also includes retrieval, tool execution, and prompt-building overhead).
    Best-effort throughout: not every provider/version populates either set
    of fields, so missing data just contributes 0 rather than raising."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "eval_count": 0,
        "eval_duration_ns": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration_ns": 0,
    }
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            totals["input_tokens"] += usage.get("input_tokens", 0) or 0
            totals["output_tokens"] += usage.get("output_tokens", 0) or 0
        meta = getattr(msg, "response_metadata", None) or {}
        totals["eval_count"] += meta.get("eval_count", 0) or 0
        totals["eval_duration_ns"] += meta.get("eval_duration", 0) or 0
        totals["prompt_eval_count"] += meta.get("prompt_eval_count", 0) or 0
        totals["prompt_eval_duration_ns"] += meta.get("prompt_eval_duration", 0) or 0
    return totals


def generation_tokens_per_second(token_usage: dict[str, int]) -> Optional[float]:
    """Real generation throughput from Ollama's own `eval_count`/
    `eval_duration`, when available. None (not 0) when the provider didn't
    report it, so callers can distinguish "we don't know" from "it was
    zero"."""
    duration_ns = token_usage.get("eval_duration_ns", 0)
    count = token_usage.get("eval_count", 0)
    if not duration_ns or not count:
        return None
    return count / (duration_ns / 1e9)


def prompt_tokens_per_second(token_usage: dict[str, int]) -> Optional[float]:
    """Real prefill throughput from Ollama's own `prompt_eval_count`/
    `prompt_eval_duration`, when available."""
    duration_ns = token_usage.get("prompt_eval_duration_ns", 0)
    count = token_usage.get("prompt_eval_count", 0)
    if not duration_ns or not count:
        return None
    return count / (duration_ns / 1e9)
