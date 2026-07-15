import pytest

from app.config import load_config
from app.data import create_structured_store


def _store():
    return create_structured_store(load_config())


def test_query_plan_data_merges_customer_and_plan_fields():
    store = _store()
    data = store.query_plan_data("CUST-1003")
    assert data["dedicated_tam"] is True
    assert data["plan_name"] == "Enterprise Plus"


def test_query_plan_data_missing_customer_raises_key_error():
    store = _store()
    with pytest.raises(KeyError):
        store.query_plan_data("does-not-exist")


def test_list_sources_returns_ten_entries():
    store = _store()
    sources = store.list_sources()
    assert len(sources) == 10
    doc_sources = [s for s in sources if s.type == "document"]
    structured_sources = [s for s in sources if s.type == "structured"]
    assert len(doc_sources) == 8
    assert len(structured_sources) == 2
