"""Unit tests for the SQL tools: derived facts, LLM-facing metadata and input validation. No database needed.

The queries themselves, and the proof that writes raise, are in test_sql_tools_db.py (needs Postgres).
"""

import uuid
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal

import pytest
from langchain_core.messages import ToolMessage
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.db.readonly import create_readonly_engine
from app.domain.enums import ClaimStatus, IncidentType, KycStatus, PaymentStatus, PolicyStatus, ProductType
from app.tools.records import (
    MAX_CLAIMS_SHOWN,
    MatchReason,
    age_on,
    build_claim_history,
    build_customer_profile,
    build_payment_history,
    build_policy_status,
    build_similar_claims,
    cover_in_force,
)
from app.tools.sql_tools import build_sql_tools

START, END = date(2026, 1, 1), date(2026, 12, 31)
SQL_TOOL_NAMES = {
    "get_policy_status", "get_claim_history", "get_customer_profile", "check_similar_claims", "get_payment_history",
}  # fmt: skip


def _policy_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "policy_number": "MOT-2026-000001",
        "product_type": ProductType.MOTOR,
        "status": PolicyStatus.ACTIVE,
        "start_date": START,
        "end_date": END,
        "lapsed_on": None,
        "sum_insured": Decimal("500000.00"),
        "deductible": Decimal("5000.00"),
        "policy_document": "Motor_Policy.pdf",
    }
    return row | overrides


# ---- policy status --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("on_date", "lapsed_on", "status", "expected"),
    [
        (date(2026, 3, 1), None, PolicyStatus.ACTIVE, True),
        (START, None, PolicyStatus.ACTIVE, True),  # first day of cover
        (END, None, PolicyStatus.EXPIRED, True),  # last day is inclusive; "expired" today doesn't matter
        (date(2025, 12, 31), None, PolicyStatus.ACTIVE, False),  # before start
        (date(2027, 1, 1), None, PolicyStatus.EXPIRED, False),  # after end
        (date(2026, 5, 31), date(2026, 6, 1), PolicyStatus.LAPSED, True),  # day before the lapse
        (date(2026, 6, 1), date(2026, 6, 1), PolicyStatus.LAPSED, False),  # lapse day has no cover
        (date(2026, 3, 1), None, PolicyStatus.CANCELLED, None),  # cancellation date unknown: don't guess
        (date(2027, 3, 1), None, PolicyStatus.CANCELLED, False),  # outside the term regardless
    ],
)
def test_cover_in_force(on_date: date, lapsed_on: date | None, status: PolicyStatus, expected: bool | None) -> None:
    assert cover_in_force(START, END, lapsed_on, status, on_date) is expected


def test_policy_status_computes_days_since_start_for_the_model() -> None:
    result = build_policy_status(_policy_row(), date(2026, 1, 13))
    assert result.in_force_on_date is True and result.days_since_start == 12 and result.note is None
    assert '"sum_insured":"500000.00"' in result.model_dump_json()  # exact decimal string, never a float


def test_policy_status_without_date_returns_the_record_only() -> None:
    result = build_policy_status(_policy_row(), None)
    assert result.in_force_on_date is None and result.days_since_start is None


def test_cancelled_policy_says_escalate_instead_of_guessing() -> None:
    result = build_policy_status(_policy_row(status=PolicyStatus.CANCELLED), date(2026, 3, 1))
    assert result.in_force_on_date is None and "Escalate" in (result.note or "")


# ---- payment history ------------------------------------------------------------------------------------------


def test_payment_history_counts_and_days_late() -> None:
    payments = [
        # Deliberately out of order: the builder sorts by installment number.
        {"installment_number": 3, "due_date": date(2026, 7, 1), "paid_date": None,
         "amount": Decimal("3125.00"), "status": PaymentStatus.MISSED},
        {"installment_number": 1, "due_date": date(2026, 1, 1), "paid_date": date(2025, 12, 28),
         "amount": Decimal("3125.00"), "status": PaymentStatus.PAID},
        {"installment_number": 2, "due_date": date(2026, 4, 1), "paid_date": date(2026, 4, 15),
         "amount": Decimal("3125.01"), "status": PaymentStatus.LATE},
    ]  # fmt: skip
    policy = {"policy_number": "HLT-2026-000002", "status": PolicyStatus.LAPSED, "lapsed_on": date(2026, 7, 31)}
    result = build_payment_history(policy, payments, on_date=date(2026, 8, 5))
    assert [item.installment_number for item in result.installments] == [1, 2, 3]
    assert [item.days_late for item in result.installments] == [0, 14, None]
    assert (result.paid_on_time, result.paid_late, result.missed) == (1, 1, 1)
    assert result.total_paid == Decimal("6250.01")
    assert result.first_missed_due_date == date(2026, 7, 1)
    assert result.missed_on_or_before_date == 1
    assert build_payment_history(policy, payments, on_date=date(2026, 6, 30)).missed_on_or_before_date == 0


def test_payment_history_with_no_installments() -> None:
    policy = {"policy_number": "HLT-2026-000002", "status": PolicyStatus.ACTIVE, "lapsed_on": None}
    result = build_payment_history(policy, [], on_date=None)
    assert result.installments_billed == 0 and result.total_paid == Decimal("0.00")
    assert result.missed_on_or_before_date is None


# ---- customer profile -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("on_date", "expected"),
    [(date(2026, 5, 9), 35), (date(2026, 5, 10), 36), (date(2027, 2, 28), 36)],
)
def test_age_counts_the_birthday_itself(on_date: date, expected: int) -> None:
    assert age_on(date(1990, 5, 10), on_date) == expected


def test_customer_profile_has_no_personal_details() -> None:
    customer = {"customer_ref": "CUS-00007", "kyc_status": KycStatus.PENDING, "date_of_birth": date(1990, 5, 10)}
    policies = [
        {"policy_number": "HOM-2026-000010", "product_type": ProductType.HOME, "status": PolicyStatus.ACTIVE,
         "start_date": date(2026, 2, 1), "end_date": date(2027, 1, 31)},
        {"policy_number": "MOT-2024-000003", "product_type": ProductType.MOTOR, "status": PolicyStatus.EXPIRED,
         "start_date": date(2024, 3, 1), "end_date": date(2025, 2, 28)},
    ]  # fmt: skip
    result = build_customer_profile(customer, policies, today=date(2026, 9, 1))
    assert result.customer_since == date(2024, 3, 1) and result.active_policies == 1 and result.age_years == 36
    assert set(result.model_dump()) == {
        "customer_ref", "kyc_status", "age_years", "customer_since", "policies", "active_policies",
    }  # fmt: skip


# ---- claim history --------------------------------------------------------------------------------------------


def _history_row(number: int, incident_date: date, status: ClaimStatus = ClaimStatus.APPROVED) -> dict[str, object]:
    return {
        "claim_number": f"CLM-2026-{number:06d}",
        "policy_number": "HOM-2025-000004",
        "product_type": ProductType.HOME,
        "incident_type": IncidentType.WATER_DAMAGE,
        "incident_date": incident_date,
        "claimed_amount": Decimal("10000.10"),
        "status": status,
        "final_outcome": None,
    }


def test_claim_history_window_matches_rule_r04() -> None:
    reference = date(2026, 9, 1)
    rows = [
        _history_row(1, date(2025, 9, 1)),  # exactly 365 days before: inside the window, as in R04
        _history_row(2, date(2025, 8, 31)),  # 366 days before: outside
        _history_row(3, date(2026, 9, 10), ClaimStatus.REJECTED),  # after the reference: listed, not counted
        _history_row(4, date(2026, 6, 1)),
    ]
    result = build_claim_history("CUS-00004", rows, reference, "CLM-2026-999999")
    assert result.other_claims_in_prior_12_months == 2
    assert result.total_claims == 4 and result.rejected_claims == 1
    assert result.total_claimed_amount == Decimal("40000.40")
    assert [claim.claim_number[-1] for claim in result.claims] == ["3", "4", "1", "2"]  # newest first
    assert result.claims[0].days_before_reference == -9


def test_claim_history_lists_only_the_newest_claims_but_counts_all() -> None:
    rows = [_history_row(n, date(2020, 1, 1 + n)) for n in range(MAX_CLAIMS_SHOWN + 5)]
    result = build_claim_history("CUS-00004", rows, date(2026, 9, 1), None)
    assert len(result.claims) == MAX_CLAIMS_SHOWN and result.claims_omitted == 5 and result.total_claims == 25


# ---- similar claims -------------------------------------------------------------------------------------------

CUSTOMER, OTHER_CUSTOMER = uuid.uuid4(), uuid.uuid4()
POLICY, OTHER_POLICY = uuid.uuid4(), uuid.uuid4()
TARGET = {
    "claim_id": uuid.uuid4(),
    "claim_number": "CLM-2026-000100",
    "policy_id": POLICY,
    "customer_id": CUSTOMER,
    "incident_type": IncidentType.COLLISION,
    "incident_date": date(2026, 8, 1),
    "claimed_amount": Decimal("40000.00"),
}


def _candidate(number: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "claim_number": f"CLM-2026-{number:06d}",
        "policy_id": POLICY,
        "policy_number": "MOT-2025-000001",
        "customer_id": CUSTOMER,
        "incident_type": IncidentType.COLLISION,
        "incident_date": date(2026, 8, 1),
        "claimed_amount": Decimal("40000.00"),
        "status": ClaimStatus.APPROVED,
    }
    return row | overrides


def test_similar_claims_classification_and_ranking() -> None:
    other = {"customer_id": OTHER_CUSTOMER, "policy_id": OTHER_POLICY, "policy_number": "MOT-2025-000009"}
    candidates = [
        _candidate(1, incident_date=date(2026, 8, 20)),  # same customer, 19 days apart
        _candidate(2),  # exact duplicate
        _candidate(3, **other, incident_date=date(2026, 8, 3), claimed_amount=Decimal("42000.00")),  # +5.0%: in
        _candidate(4, **other, incident_date=date(2026, 8, 3), claimed_amount=Decimal("42000.01")),  # just over 5%
        _candidate(5, **other, incident_date=date(2026, 8, 5)),  # 4 days apart: outside the other-customer window
        _candidate(6, **other, incident_type=IncidentType.THEFT),  # different incident type
        _candidate(7, incident_date=date(2026, 9, 1)),  # same customer, 31 days apart
    ]
    result = build_similar_claims(TARGET, candidates)
    assert [m.claim_number[-1] for m in result.matches] == ["2", "3", "1"]
    assert result.matches[0].match_reasons == (MatchReason.EXACT_DUPLICATE, MatchReason.SAME_CUSTOMER_NEAR_DATE)
    assert result.matches[1].match_reasons == (MatchReason.OTHER_CUSTOMER_SAME_PATTERN,)
    assert result.matches[1].amount_difference_pct == Decimal("5.0") and result.matches[1].days_apart == 2
    assert (result.exact_duplicates, result.same_customer_matches, result.other_customer_matches) == (1, 2, 1)


def test_no_candidates_means_an_explicit_empty_result() -> None:
    result = build_similar_claims(TARGET, [])
    assert result.matches == () and result.exact_duplicates == 0


# ---- tools: metadata and validation -------------------------------------------------------------------------


class _UntouchableDatabase:
    """Fails the test if a tool reaches the database; used to prove validation happens first."""

    @asynccontextmanager
    async def connect(self):  # type: ignore[no-untyped-def]
        raise AssertionError("the database must not be queried for invalid input")
        yield  # pragma: no cover — makes this an async generator


@pytest.fixture
def tools() -> dict[str, object]:
    return {t.name: t for t in build_sql_tools(_UntouchableDatabase(), clock=lambda: date(2026, 9, 1))}  # type: ignore[arg-type]


def test_tools_have_spec_names_and_llm_facing_descriptions(tools: dict[str, object]) -> None:
    assert set(tools) == SQL_TOOL_NAMES
    for name, sql_tool in tools.items():
        description = " ".join(sql_tool.description.split())  # type: ignore[attr-defined]
        assert "Use this" in description and "Do NOT" in description, name


def test_tool_arguments_are_typed_for_the_model(tools: dict[str, object]) -> None:
    schema = tools["get_policy_status"].args_schema.model_json_schema()  # type: ignore[attr-defined]
    assert schema["required"] == ["policy_number"]
    assert "MOT-2025-000123" in schema["properties"]["policy_number"]["description"]
    assert any(option.get("format") == "date" for option in schema["properties"]["on_date"]["anyOf"])


def _call(name: str, args: dict[str, object]) -> dict[str, object]:
    return {"type": "tool_call", "id": "call-1", "name": name, "args": args}


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_policy_status", {"policy_number": "MOT-2025-000001' OR '1'='1"}),
        ("get_payment_history", {"policy_number": "all policies"}),
        ("get_customer_profile", {"policy_number": "MOT-2025-1"}),
        ("check_similar_claims", {"claim_number": "CLM-2026-000001; DROP TABLE claims"}),
        ("get_claim_history", {"policy_number": "MOT-2025-000001", "exclude_claim_number": "latest"}),
    ],
)
async def test_invalid_identifiers_return_an_error_to_the_model_without_querying(
    tools: dict[str, object], name: str, args: dict[str, object]
) -> None:
    message = await tools[name].ainvoke(_call(name, args))  # type: ignore[attr-defined]
    assert isinstance(message, ToolMessage) and message.status == "error"
    assert "is not a valid" in message.content and "Do not invent one" in message.content


async def test_malformed_arguments_are_reported_back_not_raised(tools: dict[str, object]) -> None:
    call = _call("get_policy_status", {"policy_number": "MOT-2025-000001", "on_date": "last Tuesday"})
    message = await tools["get_policy_status"].ainvoke(call)  # type: ignore[attr-defined]
    assert isinstance(message, ToolMessage)
    assert "Invalid tool arguments" in message.content and "on_date" in message.content


# ---- configuration ----------------------------------------------------------------------------------------------


def test_tools_url_must_use_the_async_driver() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(database_url="postgresql+asyncpg://a:b@h/db", tools_database_url="postgresql+psycopg2://a:b@h/db")  # type: ignore[arg-type]


def test_readonly_engine_requires_its_own_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Otherwise Settings derives the URL from the container's AGENT_DB_USER / AGENT_DB_PASSWORD.
    for name in ("TOOLS_DATABASE_URL", "AGENT_DB_USER", "AGENT_DB_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(database_url=SecretStr("postgresql+asyncpg://a:b@h/db"), tools_database_url=None)
    with pytest.raises(RuntimeError, match="TOOLS_DATABASE_URL"):
        create_readonly_engine(settings)
