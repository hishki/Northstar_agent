"""Factory functions for the data-loading layer.

`app/factory.py` imports these two names lazily
(`from app.data import create_structured_store, create_document_store`).
"""
from __future__ import annotations

from app.config import AppConfig
from app.data.document_store import MarkdownDocumentStore
from app.data.structured_store import CsvStructuredStore
from app.interfaces import DocumentStore, StructuredStore


def create_structured_store(config: AppConfig) -> StructuredStore:
    return CsvStructuredStore(config)


def create_document_store(config: AppConfig) -> DocumentStore:
    return MarkdownDocumentStore(config)
