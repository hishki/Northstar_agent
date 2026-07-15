"""Shared pydantic models used across every module.

These are the data shapes that flow across the module boundaries defined in
`app/interfaces.py`. Every parallel workstream (data, retrieval, security,
agent) imports from here rather than defining its own overlapping types.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field


class Customer(BaseModel):
    customer_id: str
    customer_name: str
    plan_id: str
    premium_support: bool
    dedicated_tam: bool
    region: str
    contract_start: date
    contract_end: date
    post_cancel_retention_days: int
    # None means "no customer-specific override" -- the document default applies.
    migration_hours_override: Optional[int] = None


class Plan(BaseModel):
    plan_id: str
    plan_name: str
    # "custom" for Enterprise/Enterprise Plus, else a numeric string in the CSV.
    monthly_price_usd: str
    support_hours: str
    uptime_target: str
    pdf_export: bool
    saml_sso: bool
    scim: bool
    default_audit_log_days: int


class DocChunk(BaseModel):
    chunk_id: str
    source: str  # filename, e.g. "refund_policy_2026.md"
    section: str  # heading text; "" for the doc's preamble before the first heading
    text: str
    effective_date: Optional[date] = None
    published: Optional[date] = None
    version: Optional[str] = None
    # Groups documents that are different versions of "the same policy"
    # (e.g. refund_policy_2025.md / refund_policy_2026.md share doc_family
    # "refund_policy") so recency/conflict logic knows what to compare against.
    doc_family: Optional[str] = None
    suspicious: bool = False


class SearchResult(BaseModel):
    chunk: DocChunk
    score: float
    rank: int
    # Set within a doc_family group: True for the chunk with the latest
    # effective_date, False for older ones in the same family, None if the
    # chunk has no family / only one member was retrieved.
    is_newest: Optional[bool] = None
    # True when multiple family members were retrieved but recency could not
    # be resolved (e.g. missing/equal dates) -- the agent must present both.
    conflict: bool = False


class Citation(BaseModel):
    source: str
    record_id: Optional[str] = None  # chunk_id, customer_id, or plan_id
    section: Optional[str] = None
    excerpt: str


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    latency_ms: float


class SourceInfo(BaseModel):
    name: str
    type: str  # "document" | "structured"
    description: Optional[str] = None


class ToolCallRecord(BaseModel):
    """One tool invocation + its result, kept in agent state for citation
    validation -- a citation is only trusted if it points at something a
    tool actually returned during this turn."""

    tool_name: str
    input: dict[str, Any]
    output: Any
