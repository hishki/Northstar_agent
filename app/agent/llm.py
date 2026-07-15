"""Thin re-export so `app/agent/graph.py` doesn't reach into `app.factory`
directly -- keeps the agent package's own dependency surface obvious. The
actual provider dispatch (ollama/anthropic/openai) lives in
`app/factory.py::build_llm`, which is the one place that changes when
migrating to a different model provider.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import AppConfig
from app.factory import build_llm


def get_chat_model(config: AppConfig) -> BaseChatModel:
    return build_llm(config)
