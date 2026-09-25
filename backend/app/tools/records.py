"""Structured results of the read-only SQL tools, and the pure functions that derive them from database rows.

Kept apart from the queries so every derived fact (days since start, days late, window counts, similarity) is
unit-testable without a database. All date and money arithmetic happens here, in Python, never in the LLM: the
agent reads ``days_since_start: 12`` instead of subtracting two dates itself and getting it wrong.

Free-text columns (claim descriptions, customer names, contact details, city) are deliberately never returned.
Descriptions are written by claimants, so passing them to the model would open an indirect prompt-injection
channel; personal details are not needed to triage a claim, so the model never sees them (data minimisation).
Everything the model receives from these tools is an ID we generated, an enum, a date or an exact amount.
"""

from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domain.enums import (
    ClaimStatus,
    FinalOutcome,
    IncidentType,
    KycStatus,
    PaymentStatus,
    PolicyStatus,
    ProductType,
)
from app.rules.thresholds import CLAIM_FREQUENCY_WINDOW_DAYS

# A database row as returned by SQLAlchemy's ``.mappings()``, or a plain dict in unit tests.
Row = Mapping[str, Any]

# --- check_similar_claims bands --------------------------------------------------------------------------
# Two losses by one customer within a month are worth a reviewer's look: one event is sometimes split in two.
SAME_CUSTOMER_WINDOW_DAYS = 30
# Organised rings file near-identical claims within days of each other; bands this tight keep the ordinary
# coincidence of two unrelated customers with similar losses out of the results.
OTHER_CUSTOMER_WINDOW_DAYS = 3
OTHER_CUSTOMER_AMOUNT_TOLERANCE = Decimal("0.05")

# --- get_claim_history ---------------------------------------------------------------------------------------
# Same window as rule R04, so the count the agent reads agrees with what the rules engine evaluates.
CLAIM_HISTORY_WINDOW_DAYS = CLAIM_FREQUENCY_WINDOW_DAYS
# Only the newest claims are listed to keep the prompt small; the counts still cover every claim.
MAX_CLAIMS_SHOWN = 20

_ONE_DECIMAL = Decimal("0.1")


class _ToolResult(BaseModel):
    """Immutable result that rejects unknown fields, so a query and its model can't silently drift apart."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---- get_policy_status ------------------------------------------------------------------------------------


class PolicyStatusResult(_ToolResult):
    """A policy's record, optionally evaluated on one date."""

    policy_number: str
    product_type: ProductType
    status: PolicyStatus
    start_date: date
    end_date: date
    lapsed_on: date | None
    sum_insured: Decimal
    deductible: Decimal
    policy_document: str
    on_date: date | None = None
    in_force_on_date: bool | None = None
    days_since_start: int | None = None
    note: str | None = None

    def summary(self) -> str:
        """One line for the tool call log, e.g. ``MOT-2025-000001 active; in force on 2026-08-01 (day 212)``."""
        text = f"{self.policy_number} {self.status}; sum insured {self.sum_insured}"
        if self.on_date is not None:
            state = {True: "in force", False: "NOT in force", None: "cover undetermined"}[self.in_force_on_date]
            text += f"; {state} on {self.on_date} (day {self.days_since_start})"
        return text


def cover_in_force(
    start_date: date, end_date: date, lapsed_on: date | None, status: PolicyStatus, on_date: date
) -> bool | None:
    """Decide whether cover was in force on a date.

    Args:
        start_date: First day of cover.
        end_date: Last day of cover, inclusive.
        lapsed_on: First day without cover after a missed premium, if the policy lapsed.
        status: Current policy status.
        on_date: The date to evaluate, usually the incident date.

    Returns:
        True or False when the data decides it; None for a cancelled policy whose date falls inside the term,
        because the cancellation date is not recorded and guessing would be worse than saying so.
    """
    if on_date < start_date or on_date > end_date:
        return False
    if lapsed_on is not None and on_date >= lapsed_on:
        return False
    if status is PolicyStatus.CANCELLED:
        return None
    # An expired policy was still in force on any date inside its term, so "expired" alone is not a reason.
    return True


def build_policy_status(row: Row, on_date: date | None) -> PolicyStatusResult:
    """Turn a policy row into the tool result.

    Args:
        row: Columns selected by ``fetch_policy``.
        on_date: Date to evaluate cover on, or None to return the record only.

    Returns:
        The policy record, with ``in_force_on_date`` and ``days_since_start`` when ``on_date`` is given.
    """
    if on_date is None:
        return PolicyStatusResult.model_validate(dict(row))
    in_force = cover_in_force(row["start_date"], row["end_date"], row["lapsed_on"], row["status"], on_date)
    note = (
        "The policy is cancelled and its cancellation date is not recorded, so whether cover was in force on "
        "this date cannot be determined from the data. Escalate rather than assume."
        if in_force is None
        else None
    )
    return PolicyStatusResult.model_validate(
        {
            **row,
            "on_date": on_date,
            "in_force_on_date": in_force,
            "days_since_start": (on_date - row["start_date"]).days,
            "note": note,
        }
    )


# ---- get_payment_history ----------------------------------------------------------------------------------


class Installment(_ToolResult):
    """One premium installment."""

    installment_number: int
    due_date: date
    paid_date: date | None
    amount: Decimal
    status: PaymentStatus
    days_late: int | None  # 0 when paid on or before the due date; None when never paid


class PaymentHistoryResult(_ToolResult):
    """All premium installments billed on a policy, with counts."""

    policy_number: str
    policy_status: PolicyStatus
    lapsed_on: date | None
    installments: tuple[Installment, ...]
    installments_billed: int
    paid_on_time: int
    paid_late: int
    missed: int
    total_paid: Decimal
    first_missed_due_date: date | None
    on_date: date | None = None
    missed_on_or_before_date: int | None = None

    def summary(self) -> str:
        """One line for the tool call log."""
        return (
            f"{self.installments_billed} installments: {self.paid_on_time} on time, {self.paid_late} late, "
            f"{self.missed} missed"
        )


def _days_late(due_date: date, paid_date: date | None) -> int | None:
    """Days between due and payment, floored at zero; None if unpaid."""
    if paid_date is None:
        return None
    return max(0, (paid_date - due_date).days)


def build_payment_history(policy: Row, payments: Iterable[Row], on_date: date | None) -> PaymentHistoryResult:
    """Summarise a policy's premium installments.

    Args:
        policy: Row with ``policy_number``, ``status`` and ``lapsed_on``.
        payments: Rows selected by ``fetch_payments``.
        on_date: Optional date; installments missed with a due date on or before it are counted separately.

    Returns:
        Every installment in order, plus on-time, late and missed counts and the exact total paid.
    """
    installments = tuple(
        sorted(
            (
                Installment(
                    installment_number=payment["installment_number"],
                    due_date=payment["due_date"],
                    paid_date=payment["paid_date"],
                    amount=payment["amount"],
                    status=payment["status"],
                    days_late=_days_late(payment["due_date"], payment["paid_date"]),
                )
                for payment in payments
            ),
            key=lambda installment: installment.installment_number,
        )
    )
    missed = [item for item in installments if item.status is PaymentStatus.MISSED]
    return PaymentHistoryResult(
        policy_number=policy["policy_number"],
        policy_status=policy["status"],
        lapsed_on=policy["lapsed_on"],
        installments=installments,
        installments_billed=len(installments),
        paid_on_time=sum(1 for item in installments if item.status is PaymentStatus.PAID),
        paid_late=sum(1 for item in installments if item.status is PaymentStatus.LATE),
        missed=len(missed),
        # Decimal start value: sum() of an empty sequence would otherwise return the int 0.
        total_paid=sum((item.amount for item in installments if item.paid_date is not None), Decimal("0.00")),
        first_missed_due_date=min((item.due_date for item in missed), default=None),
        on_date=on_date,
        missed_on_or_before_date=None if on_date is None else sum(1 for item in missed if item.due_date <= on_date),
    )


# ---- get_customer_profile ---------------------------------------------------------------------------------


class CustomerPolicy(_ToolResult):
    """One of the customer's policies."""

    policy_number: str
    product_type: ProductType
    status: PolicyStatus
    start_date: date
    end_date: date


class CustomerProfileResult(_ToolResult):
    """Verification and relationship facts about a policyholder. No names or contact details, by design."""

    customer_ref: str
    kyc_status: KycStatus
    age_years: int
    customer_since: date
    policies: tuple[CustomerPolicy, ...]
    active_policies: int

    def summary(self) -> str:
        """One line for the tool call log."""
        return (
            f"{self.customer_ref}: KYC {self.kyc_status}, customer since {self.customer_since}, "
            f"{self.active_policies} active of {len(self.policies)} policies"
        )


def age_on(date_of_birth: date, on_date: date) -> int:
    """Whole years of age on a date.

    Args:
        date_of_birth: Birth date.
        on_date: Date to measure age on.

    Returns:
        Completed years; the birthday itself counts as the new year.
    """
    # Tuple comparison subtracts one year when the birthday hasn't come round yet this year.
    return on_date.year - date_of_birth.year - ((on_date.month, on_date.day) < (date_of_birth.month, date_of_birth.day))


def build_customer_profile(customer: Row, policies: Iterable[Row], today: date) -> CustomerProfileResult:
    """Build a customer's profile.

    Args:
        customer: Row with ``customer_ref``, ``kyc_status`` and ``date_of_birth``.
        policies: Rows selected by ``fetch_customer_policies``; at least one, since lookup is by policy.
        today: Date to compute age on.

    Returns:
        The profile, with policies ordered by start date.
    """
    items = tuple(sorted((CustomerPolicy.model_validate(dict(row)) for row in policies), key=lambda p: p.start_date))
    return CustomerProfileResult(
        customer_ref=customer["customer_ref"],
        kyc_status=customer["kyc_status"],
        age_years=age_on(customer["date_of_birth"], today),
        customer_since=items[0].start_date,
        policies=items,
        active_policies=sum(1 for item in items if item.status is PolicyStatus.ACTIVE),
    )


# ---- get_claim_history ------------------------------------------------------------------------------------


class PastClaim(_ToolResult):
    """One of the customer's other claims."""

    claim_number: str
    policy_number: str
    product_type: ProductType
    incident_type: IncidentType
    incident_date: date
    claimed_amount: Decimal
    status: ClaimStatus
    final_outcome: FinalOutcome | None
    days_before_reference: int  # negative when the incident is after the reference date


class ClaimHistoryResult(_ToolResult):
    """A customer's claims across all their policies."""

    customer_ref: str
    reference_date: date
    excluded_claim_number: str | None
    total_claims: int
    other_claims_in_prior_12_months: int
    rejected_claims: int
    total_claimed_amount: Decimal
    claims: tuple[PastClaim, ...]
    claims_omitted: int  # older claims left out of ``claims`` to keep the output short

    def summary(self) -> str:
        """One line for the tool call log."""
        return (
            f"{self.total_claims} other claims; {self.other_claims_in_prior_12_months} in the 12 months to "
            f"{self.reference_date}; {self.rejected_claims} rejected"
        )


def build_claim_history(
    customer_ref: str, rows: Iterable[Row], reference_date: date, excluded_claim_number: str | None
) -> ClaimHistoryResult:
    """Summarise a customer's claim history relative to a reference date.

    Args:
        customer_ref: The customer's reference.
        rows: Rows selected by ``fetch_customer_claims`` (the excluded claim already filtered out in SQL).
        reference_date: Usually the incident date of the claim under investigation.
        excluded_claim_number: The claim left out of the history, echoed so the model knows it was excluded.

    Returns:
        Newest-first claims with counts. The 12-month window is [reference - 365 days, reference], matching R04.
    """
    claims = sorted(
        (
            PastClaim.model_validate({**row, "days_before_reference": (reference_date - row["incident_date"]).days})
            for row in rows
        ),
        key=lambda claim: (claim.incident_date, claim.claim_number),
        reverse=True,
    )
    window_start = reference_date - timedelta(days=CLAIM_HISTORY_WINDOW_DAYS)
    return ClaimHistoryResult(
        customer_ref=customer_ref,
        reference_date=reference_date,
        excluded_claim_number=excluded_claim_number,
        total_claims=len(claims),
        # Incidents after the reference date are listed but not counted, as in R04: a later claim must not
        # change how an earlier one is judged.
        other_claims_in_prior_12_months=sum(1 for c in claims if window_start <= c.incident_date <= reference_date),
        rejected_claims=sum(1 for claim in claims if claim.status is ClaimStatus.REJECTED),
        total_claimed_amount=sum((claim.claimed_amount for claim in claims), Decimal("0.00")),
        claims=tuple(claims[:MAX_CLAIMS_SHOWN]),
        claims_omitted=max(0, len(claims) - MAX_CLAIMS_SHOWN),
    )


# ---- check_similar_claims ---------------------------------------------------------------------------------


class MatchReason(StrEnum):
    """Why a claim was reported as similar."""

    EXACT_DUPLICATE = "exact_duplicate"  # same policy, incident date and amount: what rule R07 checks
    SAME_CUSTOMER_NEAR_DATE = "same_customer_near_date"
    OTHER_CUSTOMER_SAME_PATTERN = "other_customer_same_pattern"


class SimilarClaim(_ToolResult):
    """A claim that resembles the one under investigation."""

    claim_number: str
    policy_number: str
    same_customer: bool
    incident_type: IncidentType
    incident_date: date
    claimed_amount: Decimal
    status: ClaimStatus
    days_apart: int
    amount_difference_pct: Decimal
    match_reasons: tuple[MatchReason, ...]


class SimilarClaimsResult(_ToolResult):
    """Claims resembling one claim, strongest matches first."""

    claim_number: str
    incident_type: IncidentType
    incident_date: date
    claimed_amount: Decimal
    exact_duplicates: int
    same_customer_matches: int
    other_customer_matches: int
    matches: tuple[SimilarClaim, ...]

    def summary(self) -> str:
        """One line for the tool call log."""
        if not self.matches:
            return "no similar claims"
        return (
            f"{len(self.matches)} similar: {self.exact_duplicates} exact duplicate(s), "
            f"{self.same_customer_matches} same customer, {self.other_customer_matches} other customers"
        )


def amount_band(amount: Decimal) -> tuple[Decimal, Decimal]:
    """The inclusive amount range another customer's claim must fall in to count as the same pattern.

    Args:
        amount: The claimed amount under investigation.

    Returns:
        (low, high), exact decimals; shared by the SQL prefilter and ``match_reasons`` so both agree.
    """
    tolerance = amount * OTHER_CUSTOMER_AMOUNT_TOLERANCE
    return amount - tolerance, amount + tolerance


def match_reasons(target: Row, candidate: Row) -> tuple[MatchReason, ...]:
    """Classify how a candidate claim resembles the target.

    Args:
        target: The claim under investigation (``policy_id``, ``customer_id``, ``incident_type``,
            ``incident_date``, ``claimed_amount``).
        candidate: Another claim with the same keys.

    Returns:
        Every reason that applies; empty when the candidate is not similar.
    """
    days_apart = abs((candidate["incident_date"] - target["incident_date"]).days)
    same_customer = candidate["customer_id"] == target["customer_id"]
    reasons = []
    if (
        candidate["policy_id"] == target["policy_id"]
        and candidate["incident_date"] == target["incident_date"]
        and candidate["claimed_amount"] == target["claimed_amount"]
    ):
        reasons.append(MatchReason.EXACT_DUPLICATE)
    if same_customer and days_apart <= SAME_CUSTOMER_WINDOW_DAYS:
        reasons.append(MatchReason.SAME_CUSTOMER_NEAR_DATE)
    low, high = amount_band(target["claimed_amount"])
    if (
        not same_customer
        and candidate["incident_type"] == target["incident_type"]
        and days_apart <= OTHER_CUSTOMER_WINDOW_DAYS
        and low <= candidate["claimed_amount"] <= high
    ):
        reasons.append(MatchReason.OTHER_CUSTOMER_SAME_PATTERN)
    return tuple(reasons)


def build_similar_claims(target: Row, candidates: Iterable[Row]) -> SimilarClaimsResult:
    """Keep the candidates that really match and rank them.

    Args:
        target: The claim under investigation, from ``fetch_claim_target``.
        candidates: Rows from ``fetch_similar_candidates``, a superset that is filtered here.

    Returns:
        Matches with exact duplicates first, then by closeness in date.
    """
    amount = target["claimed_amount"]
    matches = []
    for candidate in candidates:
        reasons = match_reasons(target, candidate)
        if not reasons:
            continue
        matches.append(
            SimilarClaim(
                claim_number=candidate["claim_number"],
                policy_number=candidate["policy_number"],
                same_customer=candidate["customer_id"] == target["customer_id"],
                incident_type=candidate["incident_type"],
                incident_date=candidate["incident_date"],
                claimed_amount=candidate["claimed_amount"],
                status=candidate["status"],
                days_apart=abs((candidate["incident_date"] - target["incident_date"]).days),
                amount_difference_pct=(abs(candidate["claimed_amount"] - amount) / amount * 100).quantize(
                    _ONE_DECIMAL, rounding=ROUND_HALF_UP
                ),
                match_reasons=reasons,
            )
        )
    matches.sort(key=lambda m: (MatchReason.EXACT_DUPLICATE not in m.match_reasons, m.days_apart, m.claim_number))
    return SimilarClaimsResult(
        claim_number=target["claim_number"],
        incident_type=target["incident_type"],
        incident_date=target["incident_date"],
        claimed_amount=amount,
        exact_duplicates=sum(1 for m in matches if MatchReason.EXACT_DUPLICATE in m.match_reasons),
        same_customer_matches=sum(1 for m in matches if m.same_customer),
        other_customer_matches=sum(1 for m in matches if not m.same_customer),
        matches=tuple(matches),
    )
