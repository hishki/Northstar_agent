"""Vector-index tests for app/retrieval/embeddings.py.

Runs entirely against the in-memory Qdrant path (no QDRANT_URL env set), per
the assignment's test contract.
"""
from __future__ import annotations

import numpy as np

from app.config import load_config
from app.retrieval.embeddings import EmbeddingIndex
from app.schemas import DocChunk


def _fixture_chunks() -> list[DocChunk]:
    return [
        DocChunk(
            chunk_id="refund.md#1",
            source="refund.md",
            section="Refunds",
            text="Monthly subscriptions may be refunded within 7 calendar days.",
        ),
        DocChunk(
            chunk_id="uptime.md#1",
            source="uptime.md",
            section="Uptime",
            text="The public service target is 99.9% monthly uptime for Business customers.",
        ),
        DocChunk(
            chunk_id="uptime.md#2",
            source="uptime.md",
            section="Regions",
            text="Northstar Cloud is available in the United States and European Union regions.",
        ),
    ]


def test_index_builds_correct_point_count(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = load_config()
    index = EmbeddingIndex(config)
    chunks = _fixture_chunks()

    index.build(chunks)

    assert index.count() == len(chunks)


def test_reindexing_is_idempotent(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = load_config()
    index = EmbeddingIndex(config)
    chunks = _fixture_chunks()

    index.build(chunks)
    index.build(chunks)  # rebuild with the same chunks

    assert index.count() == len(chunks)


def test_encoding_is_deterministic(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = load_config()
    index = EmbeddingIndex(config)

    text = "Monthly subscriptions may be refunded within 7 calendar days."
    vec1 = index.encode([text])[0]
    vec2 = index.encode([text])[0]

    assert np.allclose(vec1, vec2)


def test_search_returns_relevant_chunk(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = load_config()
    index = EmbeddingIndex(config)
    chunks = _fixture_chunks()
    index.build(chunks)

    results = index.search("How long is the refund window?", top_k=1)

    assert len(results) == 1
    assert results[0][0] == "refund.md#1"
