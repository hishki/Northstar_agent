"""Keyword-search tests for app/retrieval/bm25.py."""
from __future__ import annotations

from app.config import load_config
from app.data.document_store import MarkdownDocumentStore
from app.retrieval.bm25 import BM25Index, tokenize
from app.schemas import DocChunk


def _fixture_chunks() -> list[DocChunk]:
    return [
        DocChunk(
            chunk_id="refund.md#1",
            source="refund.md",
            section="Refunds",
            text="Monthly subscriptions may be refunded within 7 calendar days of purchase.",
        ),
        DocChunk(
            chunk_id="uptime.md#1",
            source="uptime.md",
            section="Uptime",
            text="The public service target is 99.9 percent monthly uptime for Business customers.",
        ),
        DocChunk(
            chunk_id="security.md#1",
            source="security.md",
            section="Encryption",
            text="Data at rest is encrypted using AES-256 and TLS in transit.",
        ),
    ]


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert tokenize("Refund Policy: 7-day window!") == [
        "refund",
        "policy",
        "7",
        "day",
        "window",
    ]


def test_keyword_heavy_query_ranks_relevant_chunk_first():
    index = BM25Index()
    chunks = _fixture_chunks()
    index.build(chunks)

    results = index.search("refund monthly subscription calendar days", top_k=3)

    # Only 2, not 3: security.md#1 shares zero stemmed tokens with this
    # query, so its raw BM25 score is exactly 0.0 and Fix 1 (zero-score
    # results excluded from search()) correctly drops it rather than
    # padding the list with a non-match.
    assert len(results) == 2
    top_chunk_id, top_score = results[0]
    assert top_chunk_id == "refund.md#1"
    assert top_score > 0


def test_unrelated_query_still_ranks_encryption_chunk_first():
    index = BM25Index()
    chunks = _fixture_chunks()
    index.build(chunks)

    results = index.search("encryption AES-256 TLS", top_k=3)

    assert results[0][0] == "security.md#1"


def test_empty_corpus_returns_no_results():
    index = BM25Index()
    index.build([])

    assert index.search("anything", top_k=5) == []


# --- Stemming (Fix 2): tokenize() must unify surface word-forms that broke
# the q13 eval question ("For how long are production backups retained?"
# answered by a model-issued query "backup retention period", which shared
# zero raw tokens with the correct chunk's "backed"/"backups"/"retained"). ---


def test_stemmer_unifies_retained_and_retention():
    assert set(tokenize("retained")) & set(tokenize("retention"))


def test_stemmer_unifies_backups_and_backup():
    assert set(tokenize("backups")) & set(tokenize("backup"))


def test_stemmer_does_not_over_collapse_unrelated_words():
    # These share a prefix/suffix but are not the same word family --
    # a targeted suffix stripper should leave them distinct.
    unrelated_pairs = [
        ("mention", "retention"),
        ("convention", "retention"),
        ("encryption", "retention"),
        ("security", "retention"),
        ("monthly", "month"),
    ]
    for word_a, word_b in unrelated_pairs:
        assert tokenize(word_a) != tokenize(word_b), (word_a, word_b)


# --- Fix 1: zero-score BM25 results must be excluded from search() output. ---


def test_zero_score_results_are_excluded():
    index = BM25Index()
    chunks = _fixture_chunks()
    index.build(chunks)

    # "encryption AES-256 TLS" has zero lexical overlap with refund.md#1 and
    # uptime.md#1 (no shared stemmed tokens at all) -- they must not appear.
    results = index.search("encryption AES-256 TLS", top_k=10)

    returned_ids = {chunk_id for chunk_id, _score in results}
    assert "refund.md#1" not in returned_ids
    assert "uptime.md#1" not in returned_ids
    assert all(score > 0.0 for _chunk_id, score in results)


# --- The concrete q13 regression test: BM25 alone (no embeddings, no
# reranker) must now surface the correct chunk for the exact query the
# model issued in the live eval run. ---


def test_backup_retention_query_surfaces_correct_chunk_via_bm25_alone():
    config = load_config()
    chunks = MarkdownDocumentStore(config).load_chunks()
    index = BM25Index()
    index.build(chunks)

    results = index.search("backup retention period", top_k=20)

    returned_ids = {chunk_id for chunk_id, _score in results}
    assert "security_whitepaper.md#backups" in returned_ids
