"""Concrete `Retriever`: BM25 and/or Qdrant vector search, fused via RRF.

Also applies the recency/conflict tagging pass over whatever
`search_documents` is about to return (see module docstring on
`_tag_conflicts` for the exact rule).
"""
from __future__ import annotations

from typing import Any, Optional

from app.config import AppConfig
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.reranker import Reranker
from app.schemas import DocChunk, DocumentContext, SearchResult


def _tag_conflicts(results: list[SearchResult]) -> list[SearchResult]:
    """Group results by `chunk.doc_family` (None never conflicts).

    Families with 2+ members present in `results`: the chunk with the
    strictly-latest non-null `effective_date` gets `is_newest=True`, the
    rest `is_newest=False`. If dates are missing or all-tied, every member
    of that family gets `conflict=True` and `is_newest=None` instead.
    Solo-family (or no-family) chunks get `is_newest=None, conflict=False`.
    """
    by_family: dict[str, list[int]] = {}
    for idx, result in enumerate(results):
        family = result.chunk.doc_family
        if family is None:
            continue
        by_family.setdefault(family, []).append(idx)

    for indices in by_family.values():
        if len(indices) < 2:
            continue

        dates = [results[i].chunk.effective_date for i in indices]
        if any(d is None for d in dates) or len(set(dates)) == 1:
            for i in indices:
                results[i] = results[i].model_copy(
                    update={"conflict": True, "is_newest": None}
                )
            continue

        newest_date = max(dates)
        for i in indices:
            results[i] = results[i].model_copy(
                update={"is_newest": results[i].chunk.effective_date == newest_date}
            )

    return results


class HybridRetriever:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._bm25 = BM25Index()
        self._embeddings = EmbeddingIndex(config)
        self._reranker = Reranker(config)
        self._chunks_by_id: dict[str, DocChunk] = {}
        # Chunks grouped by source file, in the same order `load_chunks()`
        # emitted them (preamble first, then "## " sections top-to-bottom)
        # -- this is document order, not retrieval-score order, so it's
        # what lets `get_document_context` find the *adjacent* section.
        self._chunks_by_source: dict[str, list[DocChunk]] = {}

    def index(self, chunks: list[DocChunk]) -> None:
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self._chunks_by_source = {}
        for chunk in chunks:
            self._chunks_by_source.setdefault(chunk.source, []).append(chunk)
        mode = self._config.retrieval.mode
        if mode in ("bm25", "hybrid"):
            self._bm25.build(chunks)
        if mode in ("embeddings", "hybrid"):
            self._embeddings.build(chunks)

    def get_document_context(self, chunk_id: str) -> Optional[DocumentContext]:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None:
            return None
        siblings = self._chunks_by_source.get(chunk.source, [])
        idx = next(i for i, c in enumerate(siblings) if c.chunk_id == chunk_id)
        previous = siblings[idx - 1] if idx > 0 else None
        next_chunk = siblings[idx + 1] if idx + 1 < len(siblings) else None
        return DocumentContext(chunk=chunk, previous=previous, next=next_chunk)

    def search_documents(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        mode = self._config.retrieval.mode
        rrf_k = self._config.retrieval.rrf_k
        reranker_cfg = self._config.retrieval.reranker
        use_reranker = mode == "hybrid" and reranker_cfg.enabled

        # Cast a wide net before top_k truncation / filtering so RRF and the
        # source filter both have enough candidates to work with. When the
        # reranker is in play it needs its own (usually wider) pool -- widen
        # candidate_pool so each per-list search already returns enough rows
        # to cover `reranker_cfg.pool_size` without a second round of queries.
        candidate_pool = max(top_k * 4, 20)
        if use_reranker:
            candidate_pool = max(candidate_pool, reranker_cfg.pool_size)

        ranked_lists: list[list[tuple[str, float]]] = []
        if mode in ("bm25", "hybrid"):
            ranked_lists.append(self._bm25.search(query, candidate_pool))
        if mode in ("embeddings", "hybrid"):
            ranked_lists.append(self._embeddings.search(query, candidate_pool))

        fused_scores: dict[str, float] = {}
        if mode == "hybrid":
            for ranked in ranked_lists:
                for rank, (chunk_id, _score) in enumerate(ranked):
                    fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                        rrf_k + rank + 1
                    )
        else:
            # Single-mode: use the raw score directly (no fusion needed).
            for ranked in ranked_lists:
                for chunk_id, score in ranked:
                    fused_scores[chunk_id] = score

        # When the reranker is enabled, rerank the union of BM25 +
        # embeddings candidates (each list truncated to `pool_size`, deduped
        # by chunk_id) *before* the final top_k cut below, so a chunk RRF
        # under-ranked (or dropped entirely from the fused top-k) still has
        # a chance to be promoted by a smarter relevance judgment. This has
        # to happen ahead of the RRF-based `sorted(...)` truncation -- once
        # that's cut to top_k, candidates outside it are already gone.
        ordered: list[tuple[str, float]] | None = None
        if use_reranker:
            pool_ids: dict[str, None] = {}
            for ranked in ranked_lists:
                for chunk_id, _score in ranked[: reranker_cfg.pool_size]:
                    pool_ids[chunk_id] = None
            candidates = [
                (chunk_id, self._chunks_by_id[chunk_id])
                for chunk_id in pool_ids
                if chunk_id in self._chunks_by_id
            ]
            reranked = self._reranker.rerank(query, candidates)
            if reranked is not None:
                ordered = reranked

        if ordered is None:
            # Plain RRF ordering: reranker disabled, not applicable (single
            # mode), or the reranker failed to load/score (best-effort
            # fallback -- see `Reranker.rerank`'s docstring).
            ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)

        source_filter = (filters or {}).get("source")
        results: list[SearchResult] = []
        for chunk_id, score in ordered:
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            if source_filter is not None and chunk.source != source_filter:
                continue
            results.append(SearchResult(chunk=chunk, score=score, rank=len(results)))
            if len(results) >= top_k:
                break

        return _tag_conflicts(results)
