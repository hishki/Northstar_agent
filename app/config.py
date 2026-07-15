"""Loads config/default.yaml (or $CONFIG_PATH) into typed settings.

Technology choices (which LLM provider, which vector store, which retrieval
mode, ...) live in the YAML file. Secrets and connection endpoints are read
from environment variables named by the `*_env` fields, so the same YAML
works unchanged across local dev / Docker Compose / CI.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    base_url_env: str = "OLLAMA_BASE_URL"
    temperature: float = 0.0


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    url_env: str = "QDRANT_URL"
    collection: str = "northstar_chunks"


class EmbeddingsConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many fused BM25+embeddings candidates to feed the cross-encoder
    # before the final top_k cut -- must be wider than top_k so the
    # reranker can actually promote a chunk RRF under-ranked, not just
    # re-sort whatever RRF already put in the top_k.
    pool_size: int = 20


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    top_k: int = 5
    rrf_k: int = 60
    reranker: RerankerConfig = RerankerConfig()


class ChunkingConfig(BaseModel):
    strategy: str = "heading"


class SanitizerConfig(BaseModel):
    enabled: bool = True


class DataConfig(BaseModel):
    documents_dir: str = "data/documents"
    structured_dir: str = "data/structured"


class AuthConfig(BaseModel):
    enabled: bool = True
    # Env var holding "key1:agent_id1,key2:agent_id2" pairs -- see
    # app/security/auth.py. Secrets stay in env, same convention as every
    # other *_env field in this file.
    keys_env: str = "AGENT_API_KEYS"


class ObservabilityConfig(BaseModel):
    provider: str = "langfuse"  # langfuse | none
    enabled: bool = True
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"
    host_env: str = "LANGFUSE_HOST"  # unset -> Langfuse SDK's own default (cloud.langfuse.com)


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    sanitizer: SanitizerConfig = SanitizerConfig()
    data: DataConfig = DataConfig()
    auth: AuthConfig = AuthConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    def llm_base_url(self, default: str = "http://localhost:11434") -> str:
        return os.environ.get(self.llm.base_url_env, default)

    def vector_store_url(self) -> Optional[str]:
        """None means "use an in-memory/local vector store" -- callers pass
        this straight to QdrantClient(url=...) and fall back to
        QdrantClient(":memory:") when it's None."""
        return os.environ.get(self.vector_store.url_env) or None


def load_config(path: Optional[str] = None) -> AppConfig:
    resolved = path or os.environ.get("CONFIG_PATH", "config/default.yaml")
    config_path = Path(resolved)
    raw: dict = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
    return AppConfig.model_validate(raw)
