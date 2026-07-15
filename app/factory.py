"""The only place concrete implementations get constructed.

Every function here reads `AppConfig` and returns something that satisfies a
Protocol from `app/interfaces.py`. Callers (the LangGraph orchestrator, the
API, tests) depend on the Protocol, never on a concrete library -- so
migrating technology is: implement the Protocol, add/adjust a branch here
(or in the sub-package's own `create_*` factory, see below), flip the config
field.

Convention for Phase B modules: each package exposes a single
`create_*(config: AppConfig) -> <Protocol>` factory from its `__init__.py`
(`app.data.create_structured_store` / `create_document_store`,
`app.retrieval.create_retriever`, `app.security.create_sanitizer`). That
factory owns any provider branching internal to that module (e.g. which
vector-store backend `create_retriever` talks to) so this top-level file
stays a thin, stable dispatch point and doesn't need to know retrieval- or
storage-internal details.

Imports of Phase B packages are deliberately lazy (inside the function
bodies): this file must be importable -- and `build_llm` usable -- before
those packages exist.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import AppConfig

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import-time coupling
    from langchain_core.language_models import BaseChatModel

    from app.interfaces import DocumentStore, Retriever, Sanitizer, StructuredStore


def build_llm(config: AppConfig) -> "BaseChatModel":
    provider = config.llm.provider
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.llm.model,
            base_url=config.llm_base_url(),
            temperature=config.llm.temperature,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.llm.model, temperature=config.llm.temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.llm.model, temperature=config.llm.temperature)
    raise ValueError(f"Unknown llm.provider: {provider!r}")


def build_structured_store(config: AppConfig) -> "StructuredStore":
    from app.data import create_structured_store

    return create_structured_store(config)


def build_document_store(config: AppConfig) -> "DocumentStore":
    from app.data import create_document_store

    return create_document_store(config)


def build_retriever(config: AppConfig) -> "Retriever":
    """Returns an unindexed Retriever -- caller is responsible for calling
    `.index(document_store.load_chunks())` once at startup."""
    from app.retrieval import create_retriever

    return create_retriever(config)


def build_sanitizer(config: AppConfig) -> "Sanitizer":
    from app.security import create_sanitizer

    return create_sanitizer(config)
