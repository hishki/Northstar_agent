"""CSV-backed implementation of the `StructuredStore` Protocol.

Loads `data/structured/customers.csv` and `data/structured/plans.csv` (paths
from `config.data.structured_dir`) with the stdlib `csv` module -- these
files are tiny (5 and 4 rows), so pandas would be overkill.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any, Optional

from app.config import AppConfig
from app.schemas import Customer, Plan, SourceInfo

# One-line descriptions for list_sources(), keyed by filename. Documents not
# present in this map (shouldn't happen for the shipped corpus) still get
# listed, just without a description.
_DOC_DESCRIPTIONS: dict[str, str] = {
    "data_retention.md": "Data retention and deletion rules for active and cancelled subscriptions.",
    "incident_response.md": "Incident severity levels and response/communication SLAs by plan.",
    "migration_guide.md": "Import limits and migration assistance hours for onboarding customer data.",
    "product_overview.md": "Overview of Northstar Cloud's platform capabilities, export options, and uptime targets.",
    "refund_policy_2025.md": "2025 refund policy for monthly/annual subscriptions (superseded by the 2026 policy).",
    "refund_policy_2026.md": "Current refund policy for monthly/annual subscriptions.",
    "security_whitepaper.md": "Security whitepaper covering encryption, access control, audit logs, backups, and compliance.",
    "support_handbook.md": "Support channels, support hours, and technical account management by plan.",
}

_STRUCTURED_DESCRIPTIONS: dict[str, str] = {
    "customers.csv": "Per-customer contract, plan assignment, and support attributes.",
    "plans.csv": "Per-plan pricing, support hours, and feature capabilities.",
}


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


class CsvStructuredStore:
    """Loads customers/plans CSVs once at construction and serves them from
    in-memory dicts keyed by id."""

    def __init__(self, config: AppConfig) -> None:
        self._structured_dir = Path(config.data.structured_dir)
        self._documents_dir = Path(config.data.documents_dir)
        self._customers: dict[str, Customer] = {}
        self._plans: dict[str, Plan] = {}
        self._load_plans()
        self._load_customers()

    def _load_plans(self) -> None:
        path = self._structured_dir / "plans.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                plan = Plan(
                    plan_id=row["plan_id"],
                    plan_name=row["plan_name"],
                    # "custom" for Enterprise/Enterprise Plus, else numeric-as-string.
                    monthly_price_usd=row["monthly_price_usd"],
                    support_hours=row["support_hours"],
                    uptime_target=row["uptime_target"],
                    pdf_export=_to_bool(row["pdf_export"]),
                    saml_sso=_to_bool(row["saml_sso"]),
                    scim=_to_bool(row["scim"]),
                    default_audit_log_days=int(row["default_audit_log_days"]),
                )
                self._plans[plan.plan_id] = plan

    def _load_customers(self) -> None:
        path = self._structured_dir / "customers.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                override_raw = (row.get("migration_hours_override") or "").strip()
                customer = Customer(
                    customer_id=row["customer_id"],
                    customer_name=row["customer_name"],
                    plan_id=row["plan_id"],
                    premium_support=_to_bool(row["premium_support"]),
                    dedicated_tam=_to_bool(row["dedicated_tam"]),
                    region=row["region"],
                    contract_start=date.fromisoformat(row["contract_start"]),
                    contract_end=date.fromisoformat(row["contract_end"]),
                    post_cancel_retention_days=int(row["post_cancel_retention_days"]),
                    migration_hours_override=int(override_raw) if override_raw else None,
                )
                self._customers[customer.customer_id] = customer

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        return self._customers.get(customer_id)

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        return self._plans.get(plan_id)

    def query_plan_data(
        self, customer_id: str, fields: Optional[list[str]] = None
    ) -> dict[str, Any]:
        customer = self._customers.get(customer_id)
        if customer is None:
            raise KeyError(customer_id)
        plan = self._plans.get(customer.plan_id)

        merged: dict[str, Any] = {}
        if plan is not None:
            merged.update(plan.model_dump())
        # Customer fields last so they win on the one shared key (plan_id,
        # which carries the same value from both sides anyway).
        merged.update(customer.model_dump())

        if fields is not None:
            merged = {k: v for k, v in merged.items() if k in fields}
        return merged

    def list_sources(self) -> list[SourceInfo]:
        sources: list[SourceInfo] = []
        for path in sorted(self._documents_dir.glob("*.md")):
            sources.append(
                SourceInfo(
                    name=path.name,
                    type="document",
                    description=_DOC_DESCRIPTIONS.get(path.name),
                )
            )
        for name in ("customers.csv", "plans.csv"):
            sources.append(
                SourceInfo(
                    name=name,
                    type="structured",
                    description=_STRUCTURED_DESCRIPTIONS.get(name),
                )
            )
        return sources
