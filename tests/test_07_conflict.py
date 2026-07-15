"""Recency/conflict tagging for the two refund-policy versions.

Reuses the same ad-hoc real-doc chunk loader as test_06_search_hybrid.py --
see that file's module docstring for the note on fixture duplication vs. the
real app.data chunker.
"""
from __future__ import annotations

from app.config import load_config
from app.retrieval import create_retriever
from tests.test_06_search_hybrid import _load_real_chunks


def test_2026_refund_policy_tagged_newest():
    config = load_config()
    retriever = create_retriever(config)

    all_chunks = _load_real_chunks()
    refund_chunks = [c for c in all_chunks if c.doc_family == "refund_policy"]
    assert len(refund_chunks) >= 2  # sanity: both files contributed chunks

    retriever.index(refund_chunks)

    results = retriever.search_documents("refund policy monthly subscription", top_k=5)

    by_source = {r.chunk.source: r for r in results}
    assert "refund_policy_2025.md" in by_source
    assert "refund_policy_2026.md" in by_source

    assert by_source["refund_policy_2026.md"].is_newest is True
    assert by_source["refund_policy_2025.md"].is_newest is False
    assert by_source["refund_policy_2026.md"].conflict is False
    assert by_source["refund_policy_2025.md"].conflict is False
