"""Recall check against real document content, no LLM involved.

NOTE ON TEST-FIXTURE DUPLICATION: this file re-derives DocChunks from the raw
`data/documents/*.md` files with its own ad-hoc heading-split parser (see
`_load_real_chunks` below), because the real chunker lives in `app.data`
(another engineer's parallel workstream) which this test suite must not
import. These fixture chunks are NOT the chunks the running system will
actually index -- at integration time, `app.data.create_document_store(...)
.load_chunks()` produces the real ones. Whoever wires up Phase C should index
those, not re-derive chunks like this test does.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from app.config import load_config
from app.retrieval import create_retriever
from app.schemas import DocChunk

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

# doc_family + effective_date overrides for files where the assignment
# specifies exact values (the two refund-policy versions).
_FAMILY_OVERRIDES = {
    "refund_policy_2025.md": ("refund_policy", date(2025, 3, 1)),
    "refund_policy_2026.md": ("refund_policy", date(2026, 2, 1)),
}


def _split_into_chunks(filename: str, text: str) -> list[DocChunk]:
    """Ad-hoc heading split: '## ' starts a new section; everything before
    the first '## ' (including the '# Title' line) is the "" preamble
    section. Good enough to exercise retrieval against real sentences --
    does not need to match the real chunker's behavior.
    """
    family, effective_date = _FAMILY_OVERRIDES.get(filename, (None, None))

    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in lines:
        heading_match = re.match(r"^##\s+(.*)$", line)
        if heading_match:
            sections.append((heading_match.group(1).strip(), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for idx, (heading, body_lines) in enumerate(sections):
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        chunks.append(
            DocChunk(
                chunk_id=f"{filename}#{idx}",
                source=filename,
                section=heading,
                text=body,
                effective_date=effective_date,
                doc_family=family,
            )
        )
    return chunks


def _load_real_chunks() -> list[DocChunk]:
    filenames = [
        "refund_policy_2025.md",
        "refund_policy_2026.md",
        "product_overview.md",
        "security_whitepaper.md",
        "incident_response.md",
        "support_handbook.md",
        "data_retention.md",
        "migration_guide.md",
    ]
    chunks: list[DocChunk] = []
    for filename in filenames:
        text = (DOCS_DIR / filename).read_text()
        chunks.extend(_split_into_chunks(filename, text))
    return chunks


def _sources(results) -> set[str]:
    return {r.chunk.source for r in results}


def test_refund_window_query_hits_a_refund_policy_file():
    config = load_config()
    retriever = create_retriever(config)
    retriever.index(_load_real_chunks())

    results = retriever.search_documents(
        "What is the current refund window for a monthly subscription?", top_k=5
    )

    sources = _sources(results)
    assert "refund_policy_2025.md" in sources or "refund_policy_2026.md" in sources


def test_sev1_business_query_hits_incident_response():
    config = load_config()
    retriever = create_retriever(config)
    retriever.index(_load_real_chunks())

    results = retriever.search_documents(
        "What is the SEV-1 initial response target for Business customers?", top_k=5
    )

    assert "incident_response.md" in _sources(results)


def test_backup_retention_query_hits_security_whitepaper():
    config = load_config()
    retriever = create_retriever(config)
    retriever.index(_load_real_chunks())

    results = retriever.search_documents(
        "For how long are production backups retained?", top_k=5
    )

    assert "security_whitepaper.md" in _sources(results)


def test_starter_pdf_export_query_hits_product_overview():
    config = load_config()
    retriever = create_retriever(config)
    retriever.index(_load_real_chunks())

    results = retriever.search_documents(
        "Can Starter customers export dashboards as PDF?", top_k=5
    )

    assert "product_overview.md" in _sources(results)
