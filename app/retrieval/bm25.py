"""BM25 keyword search over document chunks.

Thin wrapper around `rank_bm25.BM25Okapi`. Tokenization is deliberately
simple (lowercase + whitespace split, plus a small dependency-free suffix
stemmer -- see `_stem` below) -- good enough for the short markdown chunks
in this project and keeps the implementation dependency-free.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.schemas import DocChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_VOWELS = frozenset("aeiou")


def _has_vowel(text: str) -> bool:
    return any(ch in _VOWELS for ch in text)


# Irregular suffix replacements, checked before the generic suffix chain.
# These exist because a handful of common Latinate word families don't
# reduce to a shared root via plain suffix *removal* -- the noun form has
# an internal vowel/consonant change relative to the verb form (retain ->
# retention, detain -> detention, attain -> attention, contain ->
# contention, abstain -> abstention). All of these -- and only these --
# end in the literal string "tention", so mapping that suffix to "tain"
# unifies the family without touching unrelated words (e.g. "mention" and
# "convention" don't end in "tention" and are left untouched).
_IRREGULAR_SUFFIXES: list[tuple[str, str]] = [
    ("tention", "tain"),
]

# Generic suffix chain, longest/most-specific first. Only the first
# matching suffix is applied. "ed"/"ing"/"edly" additionally require that
# stripping the suffix leaves a root containing a vowel, so short words
# like "ring" or "red" are left alone rather than reduced to consonant
# fragments.
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ational", "ate"),
    ("ization", "ize"),
    ("ative", ""),
    ("ation", "ate"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("edly", ""),
    ("ing", ""),
    ("ies", "y"),
    ("sses", "ss"),
    ("es", ""),
    ("ed", ""),
    ("s", ""),
]

_VOWEL_GATED_SUFFIXES = frozenset({"ed", "ing", "edly"})

_MIN_ROOT_LEN = 3


def _stem(word: str) -> str:
    """Reduce a lowercase token to a crude root form.

    A small, self-contained suffix stripper (no nltk / no new dependency):
    unifies plurals and common verb/noun suffixes (backups -> backup,
    retained -> retain) plus one targeted irregular-family rule (retention
    -> retain) so BM25 doesn't miss matches purely because the query and
    the document used different surface forms of the same word.
    """
    if len(word) <= _MIN_ROOT_LEN:
        return word

    for suffix, replacement in _IRREGULAR_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[: -len(suffix)] + replacement

    for suffix, replacement in _SUFFIX_RULES:
        if not word.endswith(suffix):
            continue
        root = word[: -len(suffix)] if suffix else word
        if len(root) < _MIN_ROOT_LEN:
            continue
        if suffix in _VOWEL_GATED_SUFFIXES and not _has_vowel(root):
            continue
        return root + replacement

    return word


def tokenize(text: str) -> list[str]:
    return [_stem(tok) for tok in _TOKEN_RE.findall(text.lower())]


class BM25Index:
    """Keyword index over a fixed list of chunks, rebuilt on each `build()`."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[DocChunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        # BM25Okapi errors on an empty corpus; guard for the no-chunks case.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to `top_k` (chunk_id, score) pairs, best first.

        Zero-score entries are dropped: a score of exactly 0.0 means BM25
        found no lexical overlap at all, and letting those occupy a rank
        position would earn undeserved RRF credit downstream in
        `hybrid.py` (RRF only cares about list position, not the raw
        score, so a 0.0 match sitting at index 5 would otherwise look
        identical to a real rank-5 lexical match).
        """
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        ranked = [(chunk_id, score) for chunk_id, score in ranked if score > 0.0]
        return ranked[:top_k]
