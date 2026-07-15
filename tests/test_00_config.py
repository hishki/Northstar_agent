import os

import pytest

from app.config import load_config
from app.factory import build_llm


def test_defaults_load_from_yaml():
    config = load_config()
    assert config.llm.provider == "ollama"
    assert config.llm.model == "qwen2.5:7b-instruct"
    assert config.vector_store.provider == "qdrant"
    assert config.embeddings.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.retrieval.mode == "hybrid"
    assert config.data.documents_dir == "data/documents"
    assert config.data.structured_dir == "data/structured"


def test_llm_base_url_defaults_and_env_override(monkeypatch):
    config = load_config()
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert config.llm_base_url() == "http://localhost:11434"

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    assert config.llm_base_url() == "http://ollama:11434"


def test_vector_store_url_none_when_env_unset(monkeypatch):
    config = load_config()
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert config.vector_store_url() is None

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    assert config.vector_store_url() == "http://qdrant:6333"


def test_missing_config_path_falls_back_to_defaults():
    config = load_config(path="does/not/exist.yaml")
    assert config.llm.provider == "ollama"


def test_build_llm_dispatches_on_provider():
    config = load_config()
    llm = build_llm(config)
    assert type(llm).__name__ == "ChatOllama"


def test_build_llm_unknown_provider_raises():
    config = load_config()
    config.llm.provider = "made-up-provider"
    with pytest.raises(ValueError):
        build_llm(config)
