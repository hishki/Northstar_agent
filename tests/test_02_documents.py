from datetime import date

from app.config import load_config
from app.data.document_store import MarkdownDocumentStore


def _chunks():
    store = MarkdownDocumentStore(load_config())
    return store.load_chunks()


def test_exactly_eight_distinct_sources():
    chunks = _chunks()
    sources = {c.source for c in chunks}
    assert len(sources) == 8


def test_refund_policy_2026_effective_date_on_every_chunk():
    chunks = _chunks()
    chunks_2026 = [c for c in chunks if c.source == "refund_policy_2026.md"]
    assert chunks_2026, "expected at least one chunk from refund_policy_2026.md"
    for chunk in chunks_2026:
        assert chunk.effective_date == date(2026, 2, 1)


def test_refund_policy_files_share_doc_family():
    chunks = _chunks()
    for source in ("refund_policy_2025.md", "refund_policy_2026.md"):
        family_chunks = [c for c in chunks if c.source == source]
        assert family_chunks, f"expected chunks from {source}"
        for chunk in family_chunks:
            assert chunk.doc_family == "refund_policy"


def test_other_docs_have_no_doc_family():
    chunks = _chunks()
    other_sources = {
        "data_retention.md",
        "incident_response.md",
        "migration_guide.md",
        "product_overview.md",
        "security_whitepaper.md",
        "support_handbook.md",
    }
    for chunk in chunks:
        if chunk.source in other_sources:
            assert chunk.doc_family is None


def test_migration_guide_injection_payload_is_retrievable():
    chunks = _chunks()
    migration_chunks = [c for c in chunks if c.source == "migration_guide.md"]
    assert any("ignore" in c.text.lower() for c in migration_chunks)
    assert any(
        "ignore all previous instructions" in c.text.lower() for c in migration_chunks
    )
