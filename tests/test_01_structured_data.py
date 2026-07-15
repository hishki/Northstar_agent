from app.config import load_config
from app.data.structured_store import CsvStructuredStore


def _store() -> CsvStructuredStore:
    return CsvStructuredStore(load_config())


def test_loads_five_customers_and_four_plans():
    store = _store()
    assert len(store._customers) == 5
    assert len(store._plans) == 4


def test_cust_1003_plan_id_is_enterprise_plus():
    store = _store()
    customer = store.get_customer("CUST-1003")
    assert customer is not None
    assert customer.plan_id == "ENTERPRISE_PLUS"


def test_cust_1002_migration_hours_override_is_60():
    store = _store()
    customer = store.get_customer("CUST-1002")
    assert customer is not None
    assert customer.migration_hours_override == 60


def test_cust_1001_migration_hours_override_is_none():
    store = _store()
    customer = store.get_customer("CUST-1001")
    assert customer is not None
    assert customer.migration_hours_override is None
