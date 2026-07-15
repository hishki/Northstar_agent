"""Security module: wraps untrusted document content and flags likely
prompt-injection attempts. See `app/interfaces.py::Sanitizer` for the
contract and `app/security/sanitizer.py` for the implementation."""
from __future__ import annotations

from app.config import AppConfig
from app.interfaces import Sanitizer
from app.security.sanitizer import HeuristicSanitizer


def create_sanitizer(config: AppConfig) -> Sanitizer:
    return HeuristicSanitizer(config)
