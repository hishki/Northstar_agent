"""Orchestrator logic tested against a scripted fake chat model -- no
network calls, no Ollama required. Uses the shared test fakes (tests.fakes)
for the structured-store/sanitizer so these tests exercise the real
graph wiring (agent <-> tools loop, tool_call_log accumulation, citation
extraction + verification via the `submit_answer` tool, abstention,
loop-limit guard) without depending on any particular LLM's behavior.

Citations arrive as `submit_answer` tool-call arguments (validated by
`CitationInput`'s Pydantic schema), not a free-text block the graph has to
regex-parse -- an earlier version asked the model to end its answer with a
"Citations:\\n- source: ...; section: ...; excerpt: ..." block and parsed
that; running it against the live qwen2.5:7b-instruct model surfaced real
failures (some answers had zero citation lines, others invented a different
format) even when the model had retrieved and used the right evidence. The
interleaved/malformed-block failure modes that motivated that parser are
structurally impossible now -- there's no free text to misparse.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import _REMIND_MESSAGE, MAX_AGENT_LOOPS, build_agent_graph
from app.config import load_config
from tests.fakes import FakeSanitizer, FakeStructuredStore
from app.retrieval import create_retriever
from app.schemas import DocChunk


class _FakeStructuredRunnable:
    """What `.with_structured_output(...)` returns: something with its own
    scripted `.invoke()` queue, separate from the tool-calling one."""

    def __init__(self, parent: "FakeChatModel"):
        self._parent = parent

    def invoke(self, messages):
        self._parent.structured_invocations.append(messages)
        assert self._parent._structured_responses, "FakeChatModel ran out of scripted structured responses"
        response = self._parent._structured_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    """Minimal duck-typed stand-in for a LangChain BaseChatModel: supports
    `.bind_tools()` (returns self, ignores the schema), `.invoke()` (pops
    the next scripted tool-calling response), and `.with_structured_output()`
    (returns a runnable popping from a separate scripted queue -- this is
    what `force_submit_node` uses). Records every message list each path
    was called with for assertions."""

    def __init__(self, responses: list[AIMessage], structured_responses: Optional[list] = None):
        self._responses = list(responses)
        self._structured_responses = list(structured_responses or [])
        self.invocations: list[list] = []
        self.structured_invocations: list[list] = []

    def bind_tools(self, tools):  # noqa: ARG002 - fake ignores tool schemas
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        assert self._responses, "FakeChatModel ran out of scripted responses"
        return self._responses.pop(0)

    def with_structured_output(self, schema, method="json_schema", **kwargs):  # noqa: ARG002 - fake ignores schema/method
        return _FakeStructuredRunnable(self)


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _submit(answer, citations=None, call_id="submit_1"):
    """Build the scripted final AIMessage a well-behaved model would emit:
    a `submit_answer` tool call rather than free text."""
    return _tool_call("submit_answer", {"answer": answer, "citations": citations or []}, call_id)


def _refund_retriever(config):
    retriever = create_retriever(config)
    retriever.index(
        [
            DocChunk(
                chunk_id="fake_doc.md#refund_policy",
                source="fake_doc.md",
                section="Refund Policy",
                text="Monthly refunds within 7 days.",
                effective_date=date(2026, 2, 1),
            )
        ]
    )
    return retriever


def _build_graph(responses, config=None, structured_responses=None):
    config = config or load_config()
    retriever = _refund_retriever(config)
    structured_store = FakeStructuredStore()
    sanitizer = FakeSanitizer()
    fake_llm = FakeChatModel(responses, structured_responses=structured_responses)
    graph = build_agent_graph(config, retriever, structured_store, sanitizer, llm=fake_llm)
    return graph, fake_llm


def _invoke(graph, message, conversation_id="conv-1"):
    return graph.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "tool_call_log": [],
            "loop_count": 0,
            "submitted": False,
            "remind_count": 0,
            "customer_id": None,
        },
        config={"configurable": {"thread_id": conversation_id}},
    )


def test_search_then_submit_with_verified_citation():
    search = _tool_call("search_documents", {"query": "refund window"}, "call_1")
    final = _submit(
        "The monthly refund window is 7 days.",
        [{"source": "fake_doc.md", "section": "Refund Policy", "excerpt": "Monthly refunds within 7 days."}],
    )
    graph, fake_llm = _build_graph([search, final])

    result = _invoke(graph, "What is the refund window?")

    assert result["final_answer"] == "The monthly refund window is 7 days."
    assert result["grounded"] is True
    assert len(result["final_citations"]) == 1
    assert result["final_citations"][0].source == "fake_doc.md"
    # search_documents call + submit_answer call -- no extra loop back to
    # agent after submit_answer routes straight to finalization.
    assert len(fake_llm.invocations) == 2


def test_fabricated_excerpt_is_dropped():
    search = _tool_call("search_documents", {"query": "refund"}, "call_1")
    final = _submit(
        "Refunds take 30 days.",
        [{"source": "fake_doc.md", "section": "Refund Policy", "excerpt": "Refunds take 30 days on request."}],
    )
    graph, _ = _build_graph([search, final])

    result = _invoke(graph, "What is the refund window?")

    assert result["final_citations"] == []
    assert result["grounded"] is False


def test_structured_citation_key_equals_value_format_is_verified():
    """sample_api_contract.json's own example citation is
    `"excerpt": "dedicated_tam=true"` -- confirm that exact convention
    verifies against a query_plan_data call, not just a raw JSON dump."""
    lookup = _tool_call("query_plan_data", {"customer_id": "CUST-1001"}, "call_1")
    final = _submit(
        "Acme Retail does not have a dedicated technical account manager.",
        [{"source": "customers.csv", "section": None, "excerpt": "dedicated_tam=false"}],
    )
    graph, _ = _build_graph([lookup, final])

    result = _invoke(graph, "Does Acme Retail have a dedicated TAM?")

    assert result["grounded"] is True
    assert result["final_citations"][0].source == "customers.csv"


def test_unretrieved_source_is_dropped():
    final = _submit("Something.", [{"source": "never_fetched.md", "section": None, "excerpt": "made up"}])
    graph, _ = _build_graph([final])

    result = _invoke(graph, "Anything?")

    assert result["final_citations"] == []
    assert result["grounded"] is False


def test_abstention_via_empty_citations_list():
    final = _submit("Northstar Cloud's available documentation does not address that question.", [])
    graph, _ = _build_graph([final])

    result = _invoke(graph, "Who founded Northstar Cloud?")

    assert result["final_answer"] == "Northstar Cloud's available documentation does not address that question."
    assert result["final_citations"] == []
    assert result["grounded"] is False


def test_plain_text_prompts_a_remind_retry_then_succeeds():
    """If the model ignores rule 2 and answers as a plain chat message
    instead of calling submit_answer, it gets one nudge (the `remind` node)
    rather than being silently accepted -- and if it complies on retry, the
    citation ends up properly verified, not just passed through raw."""
    plain_text = AIMessage(content="Some answer without using the tool.", tool_calls=[])
    complies = _submit("Some answer without using the tool.", [])
    graph, fake_llm = _build_graph([plain_text, complies])

    result = _invoke(graph, "Anything?")

    assert result["final_answer"] == "Some answer without using the tool."
    assert result["final_citations"] == []
    assert result["grounded"] is False
    # First call answered as plain text; the reminder is what's in the
    # second call's message history.
    assert len(fake_llm.invocations) == 2
    assert _REMIND_MESSAGE in fake_llm.invocations[1][-1].content


def test_second_miss_escalates_to_force_submit_structured_output():
    """The actual fix, not another nudge: Ollama doesn't support forcing a
    specific tool call (tool_choice is documented as ignored), so after one
    failed text nudge, the second miss escalates to `with_structured_output`
    -- a constraint Ollama does enforce, guaranteeing a real answer instead
    of hoping a second nudge lands better than the first."""
    first_miss = AIMessage(content="Some answer without using the tool.", tool_calls=[])
    second_miss = AIMessage(content="", tool_calls=[])  # e.g. went blank on the nudge, observed live
    structured_result = SimpleNamespace(
        answer="Cedar Finance has a dedicated TAM.",
        citations=[{"source": "customers.csv", "section": None, "excerpt": "dedicated_tam=true"}],
    )
    graph, fake_llm = _build_graph([first_miss, second_miss], structured_responses=[structured_result])

    result = _invoke(graph, "Does Cedar Finance have a dedicated TAM?")

    assert len(fake_llm.invocations) == 2  # agent tried twice: once plain, once after the nudge
    assert len(fake_llm.structured_invocations) == 1  # then escalated exactly once, not looped
    assert result["final_answer"] == "Cedar Finance has a dedicated TAM."
    # customers.csv was never actually retrieved in this test (no query_plan_data
    # call scripted), so the citation correctly fails verification -- this test
    # is about force_submit firing and producing a real answer, not citation
    # verification (covered elsewhere).
    assert result["final_citations"] == []


def test_force_submit_failure_falls_back_to_raw_text():
    """Even the enforced structured call could fail transport-side (not
    observed live, but a real possibility) -- confirm that degrades to the
    same raw-text fallback used elsewhere, rather than crashing the graph."""
    first_miss = AIMessage(content="", tool_calls=[])
    second_miss = AIMessage(content="I looked but couldn't confirm.", tool_calls=[])
    graph, fake_llm = _build_graph(
        [first_miss, second_miss], structured_responses=[RuntimeError("Ollama connection reset")]
    )

    result = _invoke(graph, "Anything?")

    assert len(fake_llm.structured_invocations) == 1
    assert result["final_answer"] == "I looked but couldn't confirm."
    assert result["final_citations"] == []
    assert result["grounded"] is False


def test_customer_id_is_surfaced_into_the_conversation():
    """Regression test for a real bug found live: customer_id was tracked
    in AgentState but never actually included in what the model reads, so
    it had no way to look up a specific customer's record unless the user
    literally typed the ID. Asked about "Cedar Finance" (a name) with
    customer_id=CUST-1003 set, the model skipped query_plan_data entirely
    and guessed wrong from the general policy document alone."""
    from app.agent.graph import _build_human_message

    with_id = _build_human_message("Does Cedar Finance have a dedicated TAM?", "CUST-1003")
    assert "CUST-1003" in with_id.content
    assert "Does Cedar Finance have a dedicated TAM?" in with_id.content

    without_id = _build_human_message("What is the refund window?", None)
    assert without_id.content == "What is the refund window?"


def test_fallback_answer_strips_leaked_untrusted_tags():
    """Regression test for a real bug found live: when the model skips
    submit_answer and just answers in plain chat, it can paste a raw tool
    result -- <untrusted_document_content> tags included -- straight into
    its reply. The delimiter is meant for the model, not the end user.
    force_submit is scripted to fail here so the *raw-text* fallback path
    (the last-resort safety net, not the primary fix) is what's exercised."""
    leaked = (
        'Here is what I found:\n<untrusted_document_content source="doc.md">\n'
        "The actual policy text.\n</untrusted_document_content>\nThat's the answer."
    )
    first_miss = AIMessage(content="Thinking...", tool_calls=[])
    graph, fake_llm = _build_graph(
        [first_miss, AIMessage(content=leaked, tool_calls=[])],
        structured_responses=[RuntimeError("force_submit unavailable in this test")],
    )

    result = _invoke(graph, "Anything?")

    assert "<untrusted_document_content" in leaked  # sanity: the fixture really does contain the tag
    assert "<untrusted_document_content" not in result["final_answer"]
    assert "</untrusted_document_content>" not in result["final_answer"]
    assert "The actual policy text." in result["final_answer"]
    assert "That's the answer." in result["final_answer"]


def test_fallback_extracts_answer_from_fake_submit_call_written_as_text():
    """Regression test for a real bug found live, even after the `remind`
    nudge: the model wrote out what looks like the submit_answer call as
    plain text -- submit_answer("...", [...]) -- instead of actually
    invoking the tool. It understood the instruction but expressed it as
    pseudo-code. force_submit is scripted to fail here so the raw-text
    fallback path is what's exercised; the pseudo-code should never reach
    the user, and nothing from its malformed inline "citations" is trusted."""
    fake_call = (
        'submit_answer("Cedar Finance has a dedicated TAM as part of their Enterprise Plus plan.", '
        '[{"source": "customers.csv", "excerpt": "dedicated_tam": true}])'
    )
    first_miss = AIMessage(content="Thinking...", tool_calls=[])
    graph, fake_llm = _build_graph(
        [first_miss, AIMessage(content=fake_call, tool_calls=[])],
        structured_responses=[RuntimeError("force_submit unavailable in this test")],
    )

    result = _invoke(graph, "Does Cedar Finance have a dedicated TAM?")

    assert result["final_answer"] == "Cedar Finance has a dedicated TAM as part of their Enterprise Plus plan."
    assert "submit_answer(" not in result["final_answer"]
    assert result["final_citations"] == []
    assert result["grounded"] is False


def test_fallback_searches_backward_past_blank_responses_after_remind():
    """Regression test for a real bug found live: the model answered
    correctly as plain text on its first attempt, got the `remind` nudge
    (since it didn't call submit_answer), and then went completely blank
    (empty content, no tool call) on its second attempt. force_submit is
    scripted to fail here so the raw-text fallback -- which must search
    *backward* through the transcript rather than only checking the last
    message -- is what's exercised; reading only the last message would
    throw away the perfectly good earlier answer and return nothing."""
    good_answer = AIMessage(content="Cedar Finance has a dedicated TAM as part of their Enterprise Plus plan.", tool_calls=[])
    blank = AIMessage(content="", tool_calls=[])
    graph, fake_llm = _build_graph(
        [good_answer, blank], structured_responses=[RuntimeError("force_submit unavailable in this test")]
    )

    result = _invoke(graph, "Does Cedar Finance have a dedicated TAM?")

    assert result["final_answer"] == "Cedar Finance has a dedicated TAM as part of their Enterprise Plus plan."
    assert result["final_citations"] == []
    assert result["grounded"] is False


def test_loop_limit_guard_terminates():
    # Distinct AIMessage instances per iteration -- LangChain auto-assigns
    # each message a stable `.id`, and LangGraph's `add_messages` reducer
    # treats a repeated id as an in-place update rather than an append, so
    # reusing one object here would silently corrupt message ordering (a
    # real model never emits two responses sharing an id).
    responses = [_tool_call("search_documents", {"query": "refund"}, f"call_{i}") for i in range(MAX_AGENT_LOOPS - 1)]
    # The final scripted response still requests a tool (to actually hit
    # the loop cap on this Nth agent call) but also carries text, so the
    # forced-termination fallback path has something real to surface --
    # this last requested tool call is never executed (the loop-limit route
    # skips straight to validate_citations instead of running it).
    last = AIMessage(
        content="I don't have enough information.",
        tool_calls=[{"name": "search_documents", "args": {"query": "refund"}, "id": "call_last"}],
    )
    responses.append(last)
    graph, fake_llm = _build_graph(responses)

    result = _invoke(graph, "Loop me forever")

    assert len(fake_llm.invocations) == MAX_AGENT_LOOPS
    assert result["final_answer"] == "I don't have enough information."
    assert result["grounded"] is False


def test_tool_call_log_persists_across_turns_on_same_conversation():
    """Second turn on the same conversation_id can cite evidence retrieved
    on the first turn without re-searching."""
    turn1_search = _tool_call("search_documents", {"query": "refund window"}, "call_1")
    turn1_final = _submit(
        "The monthly refund window is 7 days.",
        [{"source": "fake_doc.md", "section": "Refund Policy", "excerpt": "Monthly refunds within 7 days."}],
        call_id="submit_1",
    )
    turn2_final = _submit(
        "Enterprise contracts are governed separately, per the same policy document.",
        [{"source": "fake_doc.md", "section": "Refund Policy", "excerpt": "Monthly refunds within 7 days."}],
        call_id="submit_2",
    )
    graph, _ = _build_graph([turn1_search, turn1_final, turn2_final])

    _invoke(graph, "What is the refund window?", conversation_id="conv-followup")
    result2 = _invoke(graph, "Does that apply to enterprise customers?", conversation_id="conv-followup")

    assert result2["grounded"] is True
    assert len(result2["final_citations"]) == 1


def test_aggregate_token_usage_sums_available_metadata():
    from app.agent.graph import aggregate_token_usage

    with_usage_1 = AIMessage(content="a", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    with_usage_2 = AIMessage(content="b", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    without_usage = AIMessage(content="c")

    totals = aggregate_token_usage([with_usage_1, without_usage, with_usage_2])

    assert totals["input_tokens"] == 17
    assert totals["output_tokens"] == 8
    # No response_metadata on any of these -- Ollama-timing fields stay at 0.
    assert totals["eval_count"] == 0
    assert totals["prompt_eval_count"] == 0


def test_aggregate_token_usage_sums_ollama_generation_timing():
    """Ollama exposes real generation/prefill timing via response_metadata
    (eval_count/eval_duration/prompt_eval_count/prompt_eval_duration, in
    nanoseconds) -- this is precise, straight from the inference engine,
    unlike a coarse estimate derived from end-to-end wall-clock latency
    (which also includes retrieval and tool-execution time)."""
    from app.agent.graph import aggregate_token_usage, generation_tokens_per_second, prompt_tokens_per_second

    msg1 = AIMessage(
        content="a",
        response_metadata={"eval_count": 100, "eval_duration": 2_000_000_000, "prompt_eval_count": 50, "prompt_eval_duration": 500_000_000},
    )
    msg2 = AIMessage(
        content="b",
        response_metadata={"eval_count": 73, "eval_duration": 3_131_798_000, "prompt_eval_count": 40, "prompt_eval_duration": 255_268_000},
    )
    no_metadata = AIMessage(content="c")

    totals = aggregate_token_usage([msg1, msg2, no_metadata])

    assert totals["eval_count"] == 173
    assert totals["eval_duration_ns"] == 5_131_798_000
    assert totals["prompt_eval_count"] == 90
    assert totals["prompt_eval_duration_ns"] == 755_268_000

    gen_speed = generation_tokens_per_second(totals)
    prompt_speed = prompt_tokens_per_second(totals)
    assert gen_speed == pytest.approx(173 / 5.131798, rel=1e-6)
    assert prompt_speed == pytest.approx(90 / 0.755268, rel=1e-6)


def test_generation_tokens_per_second_none_when_unavailable():
    from app.agent.graph import generation_tokens_per_second, prompt_tokens_per_second

    assert generation_tokens_per_second({"eval_count": 0, "eval_duration_ns": 0}) is None
    assert prompt_tokens_per_second({"prompt_eval_count": 0, "prompt_eval_duration_ns": 0}) is None


def test_chat_with_trace_returns_response_log_and_usage():
    search = _tool_call("search_documents", {"query": "refund window"}, "call_1")
    search.usage_metadata = {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24}
    final = _submit(
        "The monthly refund window is 7 days.",
        [{"source": "fake_doc.md", "section": "Refund Policy", "excerpt": "Monthly refunds within 7 days."}],
    )
    final.usage_metadata = {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}

    from unittest.mock import MagicMock, patch

    config = load_config()
    retriever = create_retriever(config)  # left unindexed -- AgentRuntime.__init__ indexes it
    structured_store = FakeStructuredStore()
    sanitizer = FakeSanitizer()
    fake_llm = FakeChatModel([search, final])

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
        from app.agent.graph import AgentRuntime

        runtime = AgentRuntime(config, llm=fake_llm)

    response, tool_call_log, token_usage = runtime.chat_with_trace(
        "What is the refund window?", conversation_id="trace-1"
    )

    assert response.grounded is True
    assert response.latency_ms >= 0
    assert len(tool_call_log) == 2  # search_documents + submit_answer
    assert tool_call_log[0].tool_name == "search_documents"
    assert tool_call_log[1].tool_name == "submit_answer"
    assert token_usage["input_tokens"] == 50
    assert token_usage["output_tokens"] == 16
    # No response_metadata scripted on these fakes -- Ollama-timing fields absent.
    assert token_usage["eval_count"] == 0
