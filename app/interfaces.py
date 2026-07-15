"""Protocol contracts for every pluggable service in the system.

These are the seams for both (a) parallel development -- each module below
can be implemented independently against its Protocol, without waiting on
the others -- and (b) technology migration -- `app/factory.py` is the only
place that chooses a concrete class per `config/default.yaml`; callers only
ever depend on the Protocol.

Note on the chat model: we deliberately do NOT define our own ChatModel
Protocol. LangChain's `langchain_core.language_models.BaseChatModel` already
is that interface (`.bind_tools()`, `.invoke()`, etc.), and every provider
we might use (ChatOllama, ChatAnthropic, ChatOpenAI) subclasses it. Reinventing
a parallel Protocol would just be a worse copy of a type that already exists.
`app/factory.py::build_llm` returns a `BaseChatModel`.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from app.schemas import Customer, DocChunk, DocumentContext, Plan, SearchResult, SourceInfo


@runtime_checkable
class StructuredStore(Protocol):
    """Structured customer/plan data (`data/structured/*.csv`)."""

    def get_customer(self, customer_id: str) -> Optional[Customer]: ...

    def get_plan(self, plan_id: str) -> Optional[Plan]: ...

    def query_plan_data(
        self, customer_id: str, fields: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Return the customer's raw merged customer+plan record.

        Deliberately NOT pre-computing "effective" values where a field only
        has a default in prose documentation (e.g. migration hours) -- the
        caller (the agent) combines this with document evidence per the
        override-precedence rule in the system prompt. Fields that ARE fully
        determined by the CSVs (dedicated_tam, premium_support, retention
        days, plan capabilities) are returned as-is and need no further
        merging.

        Raises KeyError if customer_id is not found.
        """
        ...

    def list_sources(self) -> list[SourceInfo]:
        """Enumerate every document + structured file available to the agent."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Loads and chunks `data/documents/*.md` into DocChunk records."""

    def load_chunks(self) -> list[DocChunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Hybrid (or single-mode) search over document chunks."""

    def index(self, chunks: list[DocChunk]) -> None:
        """(Re)build the search index from the given chunks."""
        ...

    def search_documents(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
    ) -> list[SearchResult]: ...

    def get_document_context(self, chunk_id: str) -> Optional[DocumentContext]:
        """Fetch a chunk by ID plus its previous/next section from the same
        document, for follow-up/citation-detail lookups that need more
        context than the single chunk `search_documents` already returned."""
        ...


@runtime_checkable
class Sanitizer(Protocol):
    """Wraps retrieved document text so injected instructions are inert."""

    def wrap(self, chunk: DocChunk) -> str:
        """Return the chunk text delimited as untrusted content, safe to
        place in the model's context."""
        ...

    def is_suspicious(self, text: str) -> bool:
        """Heuristic flag only -- never used to drop content, only to mark
        it for logging/eval (prompt-injection-resistance metric)."""
        ...
