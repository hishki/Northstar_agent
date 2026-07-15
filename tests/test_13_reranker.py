"""Tests for app/retrieval/reranker.py and its wiring into HybridRetriever.

NOTE ON MODEL DOWNLOAD: the reranker tests below construct a real
`sentence_transformers.CrossEncoder`, which downloads a ~90MB model on
first run -- exactly like `tests/test_04_embeddings.py` already does for
the bi-encoder used in `app/retrieval/embeddings.py`. That file uses no
special pytest marker for the download cost, so this file follows the same
precedent (no `@pytest.mark.live` or similar) for consistency.

This is the concrete regression suite for the q13 bug: the model issued
the query "backup retention period" (its own paraphrase) and the retrieval
stack cited the wrong document (`data_retention.md`) instead of
`security_whitepaper.md#backups`. See `app/retrieval/bm25.py` (stemming +
zero-score filtering) and `app/retrieval/hybrid.py`
(`HybridRetriever.search_documents`, reranker wiring) for the fixes.
"""
from __future__ import annotations

from app.config import load_config
from app.data.document_store import MarkdownDocumentStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker

_QUERY = "backup retention period"
_EXPECTED_CHUNK_ID = "security_whitepaper.md#backups"


def _load_real_chunks(config):
    return MarkdownDocumentStore(config).load_chunks()


def _config_with_reranker_enabled(enabled: bool):
    config = load_config()
    reranker_cfg = config.retrieval.reranker.model_copy(update={"enabled": enabled})
    retrieval_cfg = config.retrieval.model_copy(update={"reranker": reranker_cfg})
    return config.model_copy(update={"retrieval": retrieval_cfg})


def test_reranker_class_scores_relevant_chunk_above_distractor(monkeypatch):
    """Direct test of `Reranker.rerank`: given the real security_whitepaper
    backups chunk (correct answer) alongside a data_retention.md distractor
    that only wins on RRF's rank-based fusion due to surface word overlap,
    a real cross-encoder should score the genuinely relevant chunk higher.
    """
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = load_config()
    chunks_by_id = {c.chunk_id: c for c in _load_real_chunks(config)}

    reranker = Reranker(config)
    candidates = [
        ("data_retention.md#deletion_requests", chunks_by_id["data_retention.md#deletion_requests"]),
        (_EXPECTED_CHUNK_ID, chunks_by_id[_EXPECTED_CHUNK_ID]),
    ]

    ranked = reranker.rerank(_QUERY, candidates)

    assert ranked is not None
    assert ranked[0][0] == _EXPECTED_CHUNK_ID


def test_hybrid_search_promotes_backup_retention_chunk_into_top5(monkeypatch):
    """The primary acceptance test for the whole q13 fix: with the reranker
    enabled and both the BM25 stemming/zero-score fixes in place, the
    correct chunk must be in the final top 5 -- not just BM25's or the
    embedding index's own ranking in isolation.
    """
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = _config_with_reranker_enabled(enabled=True)
    retriever = HybridRetriever(config)
    retriever.index(_load_real_chunks(config))

    results = retriever.search_documents(_QUERY, top_k=5)

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert _EXPECTED_CHUNK_ID in chunk_ids


def test_reranker_disabled_matches_plain_rrf_ordering(monkeypatch):
    """With the reranker off, `search_documents` must produce exactly the
    same ordering plain RRF fusion would (byte-for-byte identical to
    pre-reranker behavior) -- computed here independently of
    `HybridRetriever`'s internals, from the same BM25/embeddings indexes,
    to prove the disabled path really is unmodified RRF.
    """
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = _config_with_reranker_enabled(enabled=False)
    retriever = HybridRetriever(config)
    chunks = _load_real_chunks(config)
    retriever.index(chunks)

    top_k = 5
    results = retriever.search_documents(_QUERY, top_k=top_k)

    candidate_pool = max(top_k * 4, 20)
    bm25_ranked = retriever._bm25.search(_QUERY, candidate_pool)
    emb_ranked = retriever._embeddings.search(_QUERY, candidate_pool)

    rrf_k = config.retrieval.rrf_k
    fused: dict[str, float] = {}
    for ranked in (bm25_ranked, emb_ranked):
        for rank, (chunk_id, _score) in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    expected_order = [
        chunk_id
        for chunk_id, _score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    ][:top_k]

    assert [r.chunk.chunk_id for r in results] == expected_order


def test_reranker_load_failure_falls_back_to_rrf(monkeypatch):
    """A cross-encoder that fails to load (e.g. no network) must not crash
    retrieval -- `search_documents` should still return results via the
    plain-RRF fallback.
    """
    monkeypatch.delenv("QDRANT_URL", raising=False)

    import sentence_transformers

    def _raise_on_load(*args, **kwargs):
        raise RuntimeError("simulated: no network / model unavailable")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _raise_on_load)

    config = _config_with_reranker_enabled(enabled=True)
    retriever = HybridRetriever(config)
    retriever.index(_load_real_chunks(config))

    results = retriever.search_documents(_QUERY, top_k=5)

    assert len(results) > 0


def test_reranker_rerank_returns_empty_list_for_no_candidates():
    config = load_config()
    reranker = Reranker(config)

    assert reranker.rerank(_QUERY, []) == []


def test_search_result_carries_both_rrf_and_rerank_scores_when_enabled(monkeypatch):
    """With the reranker enabled and actually producing an ordering, every
    result must carry both scores distinctly -- not just whichever one
    happened to win -- so a caller (or a Langfuse trace of the
    search_documents tool call, see app/agent/tools.py) can see the RRF
    ranking a chunk would have gotten alongside the cross-encoder's
    judgment, not just the final effective `score`."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = _config_with_reranker_enabled(enabled=True)
    retriever = HybridRetriever(config)
    retriever.index(_load_real_chunks(config))

    results = retriever.search_documents(_QUERY, top_k=5)

    assert results, "expected at least one result"
    for r in results:
        assert isinstance(r.rrf_score, float)
        assert isinstance(r.rerank_score, float)
        # The effective `score` used for ordering must be the reranker's
        # score here, not silently the RRF one -- otherwise a caller reading
        # `score` alone would think RRF (not the reranker) picked this order.
        assert r.score == r.rerank_score


def test_search_result_rerank_score_is_none_when_reranker_disabled(monkeypatch):
    """With the reranker off, results must still carry `rrf_score` (the only
    ranking signal in play) but `rerank_score` must be None, not some stale
    or fabricated value -- None is the caller's signal that no reranking
    happened for this call."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    config = _config_with_reranker_enabled(enabled=False)
    retriever = HybridRetriever(config)
    retriever.index(_load_real_chunks(config))

    results = retriever.search_documents(_QUERY, top_k=5)

    assert results, "expected at least one result"
    for r in results:
        assert isinstance(r.rrf_score, float)
        assert r.rerank_score is None
        assert r.score == r.rrf_score
