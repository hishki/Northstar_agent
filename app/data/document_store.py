"""Markdown-backed implementation of the `DocumentStore` Protocol.

Chunking strategy (matches `config.chunking.strategy == "heading"`): split
each `.md` file on lines starting with `"## "` (level-2 headings) into one
chunk per section. Doc-level metadata (effective date / published date /
version) is parsed from the preamble -- the text before the first
level-2 heading -- via regex, and stamped onto *every* chunk from that
document so downstream conflict-resolution logic always has the date
without having to look up a separate "preamble" chunk.

Judgment call: the preamble itself (title line + metadata lines + any lead
paragraph before the first "## ") is also emitted as its own chunk, with
`section=""` and `chunk_id` suffix `#preamble`. Two of the eight docs
(`product_overview.md` in particular) have real prose content in that lead
paragraph, so dropping the preamble would silently lose retrievable text and
a valid citation target. The preamble chunk is only skipped if it would be
empty (there's no content before the first heading at all).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from app.config import AppConfig
from app.schemas import DocChunk

# Only these two files are competing versions of the same policy in this
# corpus; every other doc stands alone.
_REFUND_POLICY_FAMILY_FILES = {"refund_policy_2025.md", "refund_policy_2026.md"}
_REFUND_POLICY_FAMILY_NAME = "refund_policy"

_HEADING_PREFIX = "## "

_EFFECTIVE_DATE_RE = re.compile(
    r"^effective date:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)
_PUBLISHED_RE = re.compile(r"^published:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
# Colon is optional -- security_whitepaper.md writes "Version 3.2", no colon.
_VERSION_RE = re.compile(r"^version:?\s*(\S+)", re.IGNORECASE)


def _slugify(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", heading.strip().lower())
    return slug.strip("_")


class MarkdownDocumentStore:
    """Reads every `.md` file under `config.data.documents_dir` and splits
    it into one `DocChunk` per level-2 section (plus an optional preamble
    chunk)."""

    def __init__(self, config: AppConfig) -> None:
        self._documents_dir = Path(config.data.documents_dir)

    def load_chunks(self) -> list[DocChunk]:
        chunks: list[DocChunk] = []
        for path in sorted(self._documents_dir.glob("*.md")):
            chunks.extend(self._load_file(path))
        return chunks

    def _load_file(self, path: Path) -> list[DocChunk]:
        source = path.name
        doc_family = (
            _REFUND_POLICY_FAMILY_NAME
            if source in _REFUND_POLICY_FAMILY_FILES
            else None
        )
        lines = path.read_text(encoding="utf-8").splitlines()

        heading_idxs = [i for i, line in enumerate(lines) if line.startswith(_HEADING_PREFIX)]
        preamble_end = heading_idxs[0] if heading_idxs else len(lines)
        preamble_lines = lines[:preamble_end]

        effective_date: date | None = None
        published: date | None = None
        version: str | None = None
        for line in preamble_lines:
            stripped = line.strip()
            if (m := _EFFECTIVE_DATE_RE.match(stripped)) is not None:
                effective_date = date.fromisoformat(m.group(1))
            elif (m := _PUBLISHED_RE.match(stripped)) is not None:
                published = date.fromisoformat(m.group(1))
            elif (m := _VERSION_RE.match(stripped)) is not None:
                version = m.group(1)

        chunks: list[DocChunk] = []

        preamble_text = "\n".join(preamble_lines).strip()
        if preamble_text:
            chunks.append(
                DocChunk(
                    chunk_id=f"{source}#preamble",
                    source=source,
                    section="",
                    text=preamble_text,
                    effective_date=effective_date,
                    published=published,
                    version=version,
                    doc_family=doc_family,
                )
            )

        for n, idx in enumerate(heading_idxs):
            heading_line = lines[idx]
            section = heading_line[len(_HEADING_PREFIX):].strip()
            end = heading_idxs[n + 1] if n + 1 < len(heading_idxs) else len(lines)
            body = "\n".join(lines[idx + 1 : end]).strip()
            text = f"{heading_line}\n\n{body}".strip() if body else heading_line

            chunks.append(
                DocChunk(
                    chunk_id=f"{source}#{_slugify(section)}",
                    source=source,
                    section=section,
                    text=text,
                    effective_date=effective_date,
                    published=published,
                    version=version,
                    doc_family=doc_family,
                )
            )

        return chunks
