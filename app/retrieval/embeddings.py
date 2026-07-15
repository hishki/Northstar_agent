"""Dense vector search backed by sentence-transformers + Qdrant.

`QdrantClient(url=...)` is used when a real Qdrant service URL is configured
(`config.vector_store_url()`), otherwise an in-memory Qdrant instance is
used (`QdrantClient(location=":memory:")`) -- same code path either way, so
tests exercise real Qdrant semantics without a running service.
"""
from __future__ import annotations

from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.config import AppConfig
from app.schemas import DocChunk


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value is not None else None


class EmbeddingIndex:
    """Encodes chunk text and stores vectors + payload in a Qdrant collection."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = SentenceTransformer(config.embeddings.model)
        url = config.vector_store_url()
        if url:
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(location=":memory:")
        self._collection = config.vector_store.collection

    def encode(self, texts: list[str]):
        return self._model.encode(texts, convert_to_numpy=True)

    def build(self, chunks: list[DocChunk]) -> None:
        """(Re)create the collection and upsert all chunks. Idempotent."""
        vector_size = self._model.get_sentence_embedding_dimension()
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )
        if not chunks:
            return

        vectors = self.encode([c.text for c in chunks])
        points = [
            qmodels.PointStruct(
                id=idx,
                vector=vectors[idx].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "effective_date": _iso(chunk.effective_date),
                    "doc_family": chunk.doc_family,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to `top_k` (chunk_id, score) pairs, best first."""
        query_vector = self.encode([query])[0].tolist()
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
        )
        return [(hit.payload["chunk_id"], hit.score) for hit in hits]
