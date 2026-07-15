"""Heuristic sanitizer: wraps untrusted document text in explicit delimiters
and flags likely prompt-injection attempts.

This module never drops, redacts, or rewrites chunk content. `wrap()` always
returns the full original text (delimited or, if the sanitizer is disabled
via config, verbatim). `is_suspicious()` is a flag-only heuristic meant to
support logging / eval metrics ("prompt-injection resistance") and to give
the agent's system prompt a signal it can use to explicitly refuse -- the
refusal decision itself is made by the agent/LLM layer, not here.
"""
from __future__ import annotations

import re

from app.config import AppConfig
from app.schemas import DocChunk

# Distinctive, unambiguous delimiter tag. Deliberately not something a real
# Markdown document would plausibly contain (unlike bare "---" or "###"),
# so a document can't spoof the boundary by including the delimiter itself
# in its own prose -- the LLM is instructed (in the system prompt, outside
# this module) that only content between these exact tags is untrusted data.
_OPEN_TAG = '<untrusted_document_content source="{source}"{section_attr}>'
_CLOSE_TAG = "</untrusted_document_content>"


def _section_attr(section: str) -> str:
    if not section:
        return ""
    # Escape double quotes so a section heading can't break out of the attribute.
    escaped = section.replace('"', "&quot;")
    return f' section="{escaped}"'


# --- Direct phrase patterns -------------------------------------------------
# Case-insensitive, matched against the raw text. Kept as whole-ish phrases
# (rather than single keywords) to keep the false-positive rate low on
# ordinary policy/support prose, which routinely uses words like "instructions",
# "system", or "override" in benign contexts.
_DIRECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions",
        r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions",
        r"disregard\s+the\s+above",
        r"forget\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions",
        r"forget\s+(all\s+)?(your\s+)?(previous\s+)?instructions",
        r"reveal\s+(your|the)\s+system\s+prompt",
        r"reveal\s+(your|the)\s+(api\s*key|hidden\s+configuration|instructions|credentials)",
        r"new\s+instructions\s*:",
        r"you\s+are\s+now\b",
        r"act\s+as\s+(a|an|if)\b",
        r"ignore\s+your\s+instructions",
        r"do\s+not\s+follow\s+(the\s+|your\s+)?(previous|prior|above)?\s*instructions",
        r"stop\s+following\s+(the\s+|your\s+)?instructions",
        r"(from|instead\s+of)\s+(now\s+on\s+)?answer.{0,40}(own\s+knowledge|training\s+data)",
        r"instead\s+of\s+the\s+supplied\s+documents",
        r"ignore\s+the\s+(supplied|provided|retrieved)\s+documents",
    ]
]

# --- Proximity patterns -----------------------------------------------------
# An imperative/disclosure verb near a sensitive noun, within a short window,
# catches paraphrases of the canonical payload without hard-coding it.
_IMPERATIVE = r"(reveal|show|print|output|give|leak|disclose|expose|display)\w*"
_SENSITIVE = r"(system\s+prompt|api\s*keys?|hidden\s+configuration|hidden\s+instructions|credentials|secrets?)"
_PROXIMITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"{_IMPERATIVE}.{{0,40}}{_SENSITIVE}", re.IGNORECASE),
    re.compile(rf"{_SENSITIVE}.{{0,40}}{_IMPERATIVE}", re.IGNORECASE),
]

_ALL_PATTERNS: list[re.Pattern[str]] = _DIRECT_PATTERNS + _PROXIMITY_PATTERNS


class HeuristicSanitizer:
    """Default `Sanitizer` implementation: delimiter wrapping + keyword/regex
    injection heuristic, both gated by `config.sanitizer.enabled`."""

    def __init__(self, config: AppConfig) -> None:
        self._enabled = config.sanitizer.enabled

    def wrap(self, chunk: DocChunk) -> str:
        if not self._enabled:
            return chunk.text
        open_tag = _OPEN_TAG.format(
            source=chunk.source, section_attr=_section_attr(chunk.section)
        )
        return f"{open_tag}\n{chunk.text}\n{_CLOSE_TAG}"

    def is_suspicious(self, text: str) -> bool:
        if not self._enabled:
            return False
        return any(pattern.search(text) for pattern in _ALL_PATTERNS)
