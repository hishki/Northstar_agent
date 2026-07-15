"""Retrieval package: BM25 + Qdrant-embeddings hybrid search.

Exposes `create_retriever`, the single factory `app/factory.py::build_retriever`
depends on (see that module's docstring for the Phase B factory convention).
"""
from __future__ import annotations

from app.config import AppConfig
from app.interfaces import Retriever
from app.retrieval.hybrid import HybridRetriever


def create_retriever(config: AppConfig) -> Retriever:
    """Returns an unindexed Retriever -- caller must call `.index(chunks)`."""
    return HybridRetriever(config)
