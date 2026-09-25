"""Enumerations for claim, policy and customer states.

StrEnum values are the exact strings stored in the database and sent over the API, so they are lowercase
and must never be renamed without a migration.
"""

from enum import StrEnum


class ProductType(StrEnum):
    """Line of insurance a policy belongs to."""

    MOTOR = "motor"
    HEALTH = "health"
    HOME = "home"


class PolicyStatus(StrEnum):
    """Lifecycle state of a policy."""

    ACTIVE = "active"
    LAPSED = "lapsed"  # a premium installment was missed and the grace period ran out
    EXPIRED = "expired"  # the term ended normally
    CANCELLED = "cancelled"  # ended early at the customer's or insurer's request


class KycStatus(StrEnum):
    """Know-Your-Customer identity verification state."""

    VERIFIED = "verified"
    PENDING = "pending"
    REJECTED = "rejected"


class IncidentType(StrEnum):
    """What happened. Valid values per product are listed in ``app.domain.products``."""

    COLLISION = "collision"
    THEFT = "theft"
    VANDALISM = "vandalism"
    NATURAL_CALAMITY = "natural_calamity"
    HOSPITALISATION = "hospitalisation"
    SURGERY = "surgery"
    DAY_CARE = "day_care"
    FIRE = "fire"
    BURGLARY = "burglary"
    WATER_DAMAGE = "water_damage"


class ClaimStatus(StrEnum):
    """Where a claim is in the triage pipeline."""

    SUBMITTED = "submitted"  # stored, not yet picked up by the agent graph
    PROCESSING = "processing"  # the agent graph is running
    AWAITING_REVIEW = "awaiting_review"  # AI recommended approve/reject; a human must confirm
    ESCALATED = "escalated"  # AI could not decide safely; needs a senior reviewer
    APPROVED = "approved"  # final human decision
    REJECTED = "rejected"  # final human decision
    FAILED = "failed"  # intake could not extract valid fields after one retry


class RecommendedOutcome(StrEnum):
    """What the decision agent recommends. Always advisory, never final."""

    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class FinalOutcome(StrEnum):
    """The binding decision recorded by a human reviewer."""

    APPROVE = "approve"
    REJECT = "reject"


class PaymentStatus(StrEnum):
    """State of one premium installment."""

    PAID = "paid"
    LATE = "late"  # paid, but after the due date
    MISSED = "missed"  # never paid; triggers a lapse once the grace period ends
