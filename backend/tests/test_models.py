"""Schema-level guarantees checked without a database."""

import warnings

from sqlalchemy import DateTime, Numeric
from sqlalchemy.orm import configure_mappers

from app.db import models  # noqa: F401 — registers tables
from app.db.base import Base


def test_expected_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "customers",
        "policies",
        "premium_payments",
        "claims",
        "claim_runs",
        "audit_log",
    }


def test_money_columns_are_exact_decimals() -> None:
    money = {
        ("policies", "sum_insured"),
        ("policies", "deductible"),
        ("policies", "annual_premium"),
        ("premium_payments", "amount"),
        ("claims", "claimed_amount"),
    }
    for table, column in money:
        column_type = Base.metadata.tables[table].c[column].type
        assert isinstance(column_type, Numeric) and column_type.asdecimal, f"{table}.{column}"
        assert column_type.scale == 2, f"{table}.{column}"


def test_every_timestamp_is_timezone_aware() -> None:
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, DateTime):
                assert column.type.timezone, f"{table.name}.{column.name} must be timezone-aware"


def test_mappers_configure_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # relationship overlap warnings become test failures
        configure_mappers()
