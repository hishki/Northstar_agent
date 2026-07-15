"""The 4 tools bound to the LLM, wrapping the Retriever / StructuredStore /
Sanitizer modules built in Phase B.

Each tool returns a plain JSON-serializable Python object (not a
pre-formatted string) -- `app/agent/graph.py`'s custom tools node handles
serialization for the ToolMessage and, separately, records the raw object in
`tool_call_log` for citation validation. `search_documents` and
`get_document_context` are where `Sanitizer.wrap`/`is_suspicious` actually
get applied (this answers the ownership question flagged during Phase B:
the agent layer, right before content reaches the model, is what wraps and
flags -- not the data or retrieval layers).

`query_plan_data`'s output always includes `"source": "customers.csv"` --
this is a deliberate simplification (documented in DESIGN.md) so citation
validation has one unambiguous filename to check structured-data citations
against, even though the merged record technically draws on `plans.csv`
too. `search_documents` takes a flat `source: Optional[str]` filter (not a
generic `filters: dict`) because flat, simply-typed tool arguments are
noticeably more reliable for a small local model's tool-calling than nested
object arguments.

The model-facing descriptions below (the docstrings LangChain would
otherwise read off each function) are placeholders only -- the real,
editable text lives in `prompts/tool_descriptions.yaml` and is applied to
each tool's `.description` at the bottom of `build_tools`, so tuning a
tool's prompt doesn't require touching this file.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.agent.prompts import load_tool_descriptions
from app.interfaces import Retriever, Sanitizer, StructuredStore


class CitationInput(BaseModel):
    source: str = Field(description="Exact filename you retrieved this from, e.g. \"refund_policy_2026.md\" or \"customers.csv\".")
    section: Optional[str] = Field(default=None, description="Section name if applicable, else omit/null.")
    excerpt: str = Field(description="Short verbatim excerpt (under 200 characters) from the tool result that supports the claim.")


def build_tools(
    retriever: Retriever,
    structured_store: StructuredStore,
    sanitizer: Sanitizer,
) -> dict[str, BaseTool]:
    @tool
    def search_documents(query: str, source: Optional[str] = None, top_k: int = 5) -> list[dict]:
        """See prompts/tool_descriptions.yaml -- overridden below."""
        filters = {"source": source} if source else None
        results = retriever.search_documents(query, filters=filters, top_k=top_k)
        out = []
        for r in results:
            chunk = r.chunk
            out.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "effective_date": chunk.effective_date.isoformat() if chunk.effective_date else None,
                    "is_newest": r.is_newest,
                    "conflict": r.conflict,
                    "suspicious": sanitizer.is_suspicious(chunk.text),
                    "content": sanitizer.wrap(chunk),
                }
            )
        return out

    @tool
    def get_document_context(chunk_id: str) -> dict:
        """See prompts/tool_descriptions.yaml -- overridden below."""
        ctx = retriever.get_document_context(chunk_id)
        if ctx is None:
            return {"error": f"No chunk found with chunk_id={chunk_id!r}"}
        chunk = ctx.chunk
        # Surround the requested chunk with its neighboring section(s) from
        # the same document (document order) -- this is what makes the tool
        # actually return *more* than the single chunk `search_documents`
        # already gave the model, instead of just echoing it back.
        neighbors = [c for c in (ctx.previous, chunk, ctx.next) if c is not None]
        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "section": chunk.section,
            "effective_date": chunk.effective_date.isoformat() if chunk.effective_date else None,
            "previous_section": ctx.previous.section if ctx.previous else None,
            "next_section": ctx.next.section if ctx.next else None,
            "suspicious": any(sanitizer.is_suspicious(c.text) for c in neighbors),
            "content": "\n\n".join(sanitizer.wrap(c) for c in neighbors),
        }

    @tool
    def query_plan_data(customer_id: str) -> dict:
        """See prompts/tool_descriptions.yaml -- overridden below."""
        # No `fields` filter param exposed to the model: an earlier version
        # had one, and running live the model guessed a plausible-looking
        # but nonexistent field name ("technical_account_management"
        # instead of the real "dedicated_tam"), silently got back an empty
        # dict, and concluded there was no data at all. The full record is
        # small -- always returning everything removes that failure mode.
        try:
            data = structured_store.query_plan_data(customer_id)
        except KeyError:
            return {"error": f"No customer found with customer_id={customer_id!r}"}
        return {"source": "customers.csv", "customer_id": customer_id, "fields": data}

    @tool
    def list_sources() -> list[dict]:
        """See prompts/tool_descriptions.yaml -- overridden below."""
        return [s.model_dump() for s in structured_store.list_sources()]

    @tool
    def submit_answer(answer: str, citations: list[CitationInput]) -> dict:
        """See prompts/tool_descriptions.yaml -- overridden below."""
        return {"answer": answer, "citations": [c.model_dump() for c in citations]}

    tools = [search_documents, get_document_context, query_plan_data, list_sources, submit_answer]
    descriptions = load_tool_descriptions()
    for t in tools:
        if t.name in descriptions:
            t.description = descriptions[t.name]
    return {t.name: t for t in tools}
