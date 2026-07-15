"""LangGraph state schema for the Northstar Cloud agent.

Design notes on the two counters:

- `tool_call_log` accumulates via `operator.add` (list concatenation) and is
  intentionally allowed to persist and grow **across the whole conversation**
  (not reset each turn). Citation validation checks a citation's source
  against everything retrieved anywhere in the conversation so far, not just
  this turn -- this is what lets a follow-up ("Does that apply to enterprise
  customers?") cite evidence gathered on the previous turn without forcing a
  redundant re-search. Callers pass `tool_call_log=[]` on every `.invoke()`
  call regardless of turn number; combined with the `operator.add` reducer,
  that's a safe no-op merge against whatever is already persisted (and a
  correct initializer on the very first turn of a new conversation).
- `loop_count` has NO reducer (plain "last value wins" channel), and callers
  always pass `loop_count=0` on every `.invoke()` call. This makes it reset
  every turn while still incrementing within a turn's internal
  agent<->tools loop, guarding against a small local model looping forever
  on tool calls without ever producing a final answer.
- `submitted` (also no reducer) is set by the tools node each round to
  whether `submit_answer` was among the calls it just executed -- the
  citation-validation node runs on the very next step, instead of looping
  back to the agent, exactly when this is true. The answer/citations arrive
  as validated, typed tool-call arguments (`CitationInput` in
  `app/agent/tools.py`) rather than free text the model was asked to format
  a certain way -- small local models are far more reliable at filling in a
  typed tool schema than at following an exact prose format on every turn.
- `remind_count` (also no reducer, reset to 0 each turn) counts how many
  times the `remind` node has fired this turn. Ollama does not support
  forcing a specific tool call (`ChatOllama.bind_tools()` documents
  `tool_choice` as ignored), so a text nudge is the only cheap first
  attempt -- but observed live, a model that ignores the nudge once can
  go completely blank on further nudges too. After exactly one failed
  nudge, `_route_after_agent` escalates to `force_submit` instead of
  nudging again: a `with_structured_output(..., method="json_schema")`
  call, which Ollama *does* enforce at the grammar level -- the model
  cannot produce anything except valid `{answer, citations}` JSON.
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.schemas import Citation, ToolCallRecord


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_log: Annotated[list[ToolCallRecord], operator.add]
    loop_count: int
    submitted: bool
    remind_count: int
    customer_id: Optional[str]
    # Populated by the validate_citations node as the final step before END.
    final_answer: Optional[str]
    final_citations: Optional[list[Citation]]
    grounded: Optional[bool]
