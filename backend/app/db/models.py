"""ORM models for every table in the Data layer.

Conventions:
- Money is NUMERIC(14, 2) and maps to ``Decimal``; float rounding errors are unacceptable for payouts.
- Every timestamp is TIMESTAMP WITH TIME ZONE (see ``Base.type_annotation_map``).
- Relationships use ``lazy="raise"``: in async SQLAlchemy an implicit lazy load crashes with an obscure
  MissingGreenlet error, so we force callers to load related rows explicitly with ``selectinload``.
- Invariants that must hold no matter which code path writes are CHECK constraints, not just app logic.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domain.enums import (
    ClaimStatus,
    FinalOutcome,
    IncidentType,
    KycStatus,
    PaymentStatus,
    PolicyStatus,
    ProductType,
    RecommendedOutcome,
)

# 14 digits with 2 decimals holds values up to 999,999,999,999.99, far above any single claim.
Money = Numeric(14, 2)


def _enum_column(enum_cls: type[StrEnum], name: str) -> Enum:
    """Build a VARCHAR + CHECK constraint column type for a StrEnum.

    Native Postgres ENUM types are avoided because adding a value needs ALTER TYPE outside a transaction,
    which makes migrations awkward; a CHECK constraint gives the same safety with plain ALTER TABLE.

    Args:
        enum_cls: The StrEnum whose values are allowed.
        name: Constraint name suffix (becomes ``ck_<table>_<name>``).

    Returns:
        A SQLAlchemy Enum type storing the enum's lowercase values.
    """
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        # Store member values ("lapsed"), not member names ("LAPSED"), which is SQLAlchemy's default.
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )


class UuidPrimaryKeyMixin:
    """UUID primary key generated in Python, with a database default for raw SQL inserts."""

    # UUIDs rather than serial integers so IDs in URLs don't reveal claim volume or allow enumeration.
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )


class CreatedAtMixin:
    """Insertion timestamp set by the database clock."""

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TimestampMixin(CreatedAtMixin):
    """Insertion and last-modification timestamps."""

    # onupdate fires for ORM updates only; every write in this app goes through the ORM.
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Customer(UuidPrimaryKeyMixin, TimestampMixin, Base):
    """A policyholder."""

    __tablename__ = "customers"

    customer_ref: Mapped[str] = mapped_column(String(16), unique=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    phone: Mapped[str] = mapped_column(String(20))
    date_of_birth: Mapped[date]
    city: Mapped[str] = mapped_column(String(80))
    kyc_status: Mapped[KycStatus] = mapped_column(_enum_column(KycStatus, "kyc_status"))

    policies: Mapped[list["Policy"]] = relationship(back_populates="customer", lazy="raise")


class Policy(UuidPrimaryKeyMixin, TimestampMixin, Base):
    """An insurance contract covering one customer for one product and term."""

    __tablename__ = "policies"
    __table_args__ = (
        CheckConstraint("end_date > start_date", name="end_after_start"),
        CheckConstraint("sum_insured > 0", name="sum_insured_positive"),
        CheckConstraint("deductible >= 0", name="deductible_non_negative"),
        CheckConstraint("annual_premium > 0", name="annual_premium_positive"),
        # A lapsed policy without a lapse date would make rule R02 impossible to evaluate.
        CheckConstraint("(status = 'lapsed') = (lapsed_on IS NOT NULL)", name="lapsed_on_matches_status"),
    )

    policy_number: Mapped[str] = mapped_column(String(24), unique=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("customers.id", ondelete="RESTRICT"), index=True)
    product_type: Mapped[ProductType] = mapped_column(_enum_column(ProductType, "product_type"))
    status: Mapped[PolicyStatus] = mapped_column(_enum_column(PolicyStatus, "status"))
    start_date: Mapped[date]
    end_date: Mapped[date] = mapped_column(comment="Last day of cover, inclusive")
    lapsed_on: Mapped[date | None] = mapped_column(comment="First day without cover after a missed premium")
    sum_insured: Mapped[Decimal] = mapped_column(Money)
    deductible: Mapped[Decimal] = mapped_column(Money)
    annual_premium: Mapped[Decimal] = mapped_column(Money)
    policy_document: Mapped[str] = mapped_column(String(120), comment="Wording PDF that citations refer to")

    customer: Mapped[Customer] = relationship(back_populates="policies", lazy="raise")
    claims: Mapped[list["Claim"]] = relationship(back_populates="policy", lazy="raise")
    payments: Mapped[list["PremiumPayment"]] = relationship(back_populates="policy", lazy="raise")


class PremiumPayment(UuidPrimaryKeyMixin, CreatedAtMixin, Base):
    """One premium installment on a policy.

    Not in the original 5-table list: added because the get_payment_history tool needs real payment rows,
    and a JSONB blob would store money as untyped JSON strings.
    """

    __tablename__ = "premium_payments"
    __table_args__ = (
        UniqueConstraint("policy_id", "installment_number", name="uq_premium_payments_policy_installment"),
        CheckConstraint("amount > 0", name="amount_positive"),
        CheckConstraint("installment_number >= 1", name="installment_number_positive"),
        CheckConstraint("(status = 'missed') = (paid_date IS NULL)", name="paid_date_matches_status"),
    )

    policy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("policies.id", ondelete="CASCADE"), index=True)
    installment_number: Mapped[int] = mapped_column(SmallInteger)
    due_date: Mapped[date]
    paid_date: Mapped[date | None]
    amount: Mapped[Decimal] = mapped_column(Money)
    status: Mapped[PaymentStatus] = mapped_column(_enum_column(PaymentStatus, "status"))

    policy: Mapped[Policy] = relationship(back_populates="payments", lazy="raise")


class Claim(UuidPrimaryKeyMixin, TimestampMixin, Base):
    """A request for payment under a policy, plus the AI recommendation and the human decision."""

    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint("claimed_amount > 0", name="claimed_amount_positive"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"),
        CheckConstraint("risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)", name="risk_score_range"),
        # A final decision without a named reviewer and time would be unauditable.
        CheckConstraint(
            "final_outcome IS NULL OR (decided_by IS NOT NULL AND decided_at IS NOT NULL)",
            name="final_outcome_has_reviewer",
        ),
        # Serves both claim-history lookups and the duplicate check (R07: same policy, date and amount).
        Index("ix_claims_policy_id_incident_date", "policy_id", "incident_date"),
    )

    claim_number: Mapped[str] = mapped_column(String(24), unique=True)
    policy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("policies.id", ondelete="RESTRICT"))
    incident_type: Mapped[IncidentType] = mapped_column(_enum_column(IncidentType, "incident_type"))
    incident_date: Mapped[date]
    description: Mapped[str] = mapped_column(Text)
    claimed_amount: Mapped[Decimal] = mapped_column(Money)
    status: Mapped[ClaimStatus] = mapped_column(_enum_column(ClaimStatus, "status"), index=True)
    document_filename: Mapped[str | None] = mapped_column(String(255))
    extracted_fields: Mapped[dict[str, Any] | None] = mapped_column(JSONB, comment="Intake agent output")

    # AI recommendation, written by the decision node. Advisory only.
    recommended_outcome: Mapped[RecommendedOutcome | None] = mapped_column(
        _enum_column(RecommendedOutcome, "recommended_outcome")
    )
    # NUMERIC(4,3), not float, so the 0.65 routing threshold compares exactly.
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    risk_score: Mapped[int | None] = mapped_column(SmallInteger)
    decision_rationale: Mapped[str | None] = mapped_column(Text)

    # Binding human decision.
    final_outcome: Mapped[FinalOutcome | None] = mapped_column(_enum_column(FinalOutcome, "final_outcome"))
    decided_by: Mapped[str | None] = mapped_column(String(120))
    decided_at: Mapped[datetime | None]
    override_reason: Mapped[str | None] = mapped_column(Text)

    policy: Mapped[Policy] = relationship(back_populates="claims", lazy="raise")
    runs: Mapped[list["ClaimRun"]] = relationship(back_populates="claim", lazy="raise", order_by="ClaimRun.created_at")


# Separate index so the claims queue can sort newest-first without a full table scan.
Index("ix_claims_created_at", Claim.created_at.desc())


class ClaimRun(UuidPrimaryKeyMixin, CreatedAtMixin, Base):
    """One agent node execution for a claim. Powers the reasoning trace in the UI."""

    __tablename__ = "claim_runs"
    __table_args__ = (
        CheckConstraint("latency_ms >= 0", name="latency_ms_non_negative"),
        Index("ix_claim_runs_claim_id_created_at", "claim_id", "created_at"),
    )

    # CASCADE: runs are meaningless without their claim (claims themselves are never deleted in practice).
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"))
    agent_name: Mapped[str] = mapped_column(String(50))
    input: Mapped[dict[str, Any]] = mapped_column(JSONB)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_ms: Mapped[int] = mapped_column(Integer)

    claim: Mapped[Claim] = relationship(back_populates="runs", lazy="raise")


class AuditLog(CreatedAtMixin, Base):
    """Append-only record of every state change. A database trigger rejects UPDATE and DELETE."""

    __tablename__ = "audit_log"

    # Identity (not UUID) gives a strict insertion order; ALWAYS stops anyone from inserting a chosen ID.
    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    # RESTRICT (not SET NULL): SET NULL would issue an UPDATE, which the append-only trigger forbids anyway.
    claim_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("claims.id", ondelete="RESTRICT"), index=True)
    actor: Mapped[str] = mapped_column(String(120), comment="e.g. 'system', 'customer_portal', 'reviewer:priya.nair'")
    action: Mapped[str] = mapped_column(String(64), comment="Dotted verb, e.g. 'claim.decided'")
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    reason: Mapped[str | None] = mapped_column(Text)
