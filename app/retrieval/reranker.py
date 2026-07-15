"""Cross-encoder reranker for the fused BM25+embeddings candidate pool.

`sentence_transformers.CrossEncoder` scores a `(query, chunk_text)` pair
directly (unlike the bi-encoder in `embeddings.py`, which encodes query and
document independently and compares vectors) -- it's slower per-pair but
much better at judging genuine relevance, which is exactly what's needed
when RRF's rank-based fusion under-ranks a chunk that used different
surface wording than the query (see `hybrid.py`'s module docstring / the
q13 regression test for the motivating case).

Best-effort, matching this codebase's existing philosophy (e.g.
`aggregate_token_usage` in `app/agent/graph.py`): if the model can't be
loaded (no network, etc.) or scoring blows up for any reason, callers get
`None` back and are expected to fall back to plain RRF ordering rather than
letting a broken reranker take down retrieval entirely.

The `CrossEncoder` import and model load are both lazy (only on first
`rerank()` call) so importing this module, or constructing a `Reranker`,
never pays the model-download/load cost -- tests and callers that don't
need reranking (or run with it disabled) aren't slowed down.
"""
from __future__ import annotations

from app.config import AppConfig
from app.schemas import DocChunk


class Reranker:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = None
        self._load_failed = False

    def _get_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self._config.retrieval.reranker.model)
        except Exception:
            self._load_failed = True
            self._model = None
        return self._model

    def rerank(
        self, query: str, candidates: list[tuple[str, DocChunk]]
    ) -> list[tuple[str, float]] | None:
        """Score and re-sort `candidates` (chunk_id, chunk) pairs, best first.

        Returns `None` (never an empty list, unless `candidates` itself was
        empty) when the model can't be loaded or scoring fails, so callers
        can distinguish "reranking unavailable, fall back" from "reranked,
        happens to be empty".
        """
        if not candidates:
            return []
        model = self._get_model()
        if model is None:
            return None
        try:
            pairs = [(query, chunk.text) for _, chunk in candidates]
            scores = model.predict(pairs)
        except Exception:
            return None
        scored = list(zip((chunk_id for chunk_id, _ in candidates), scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in scored]
