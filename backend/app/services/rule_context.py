"""Build a RuleContext from database rows.

Lives in the service layer, not in app.rules, so the rules package never imports the ORM. This function does
no I/O itself: the caller loads the rows (Phase 6 rules node) and passes them in.
"""

from collections.abc import Iterable
from datetime import UTC, date

from app.db.models import Claim, Customer, Policy
from app.rules.models import ClaimFacts, CustomerFacts, PolicyFacts, PriorClaim, RuleContext


def build_rule_context(
    claim: Claim,
    policy: Policy,
    customer: Customer,
    customer_claims: Iterable[Claim],
    submitted_on: date | None = None,
) -> RuleContext:
    """Convert ORM objects into the rules engine's input.

    Args:
        claim: The claim to evaluate.
        policy: The policy the claim is made under.
        customer: The policyholder.
        customer_claims: All of the customer's claims on any policy; the evaluated claim is skipped if present.
        submitted_on: Submission date in the insurer's business timezone. Defaults to the UTC date of
            ``claim.created_at``.

    Returns:
        A validated RuleContext.

    Raises:
        ValueError: If the rows don't belong together (wrong policy or customer).
    """
    if policy.customer_id != customer.id:
        raise ValueError("policy does not belong to customer")
    return RuleContext(
        claim=ClaimFacts(
            claim_id=claim.id,
            policy_id=claim.policy_id,
            incident_date=claim.incident_date,
            claimed_amount=claim.claimed_amount,
            submitted_at=claim.created_at,
            submitted_on=submitted_on or claim.created_at.astimezone(UTC).date(),
        ),
        policy=PolicyFacts(
            policy_id=policy.id,
            start_date=policy.start_date,
            end_date=policy.end_date,
            lapsed_on=policy.lapsed_on,
            sum_insured=policy.sum_insured,
        ),
        customer=CustomerFacts(customer_id=customer.id, kyc_status=customer.kyc_status),
        other_claims=tuple(
            PriorClaim(
                claim_id=other.id,
                policy_id=other.policy_id,
                incident_date=other.incident_date,
                claimed_amount=other.claimed_amount,
                submitted_at=other.created_at,
            )
            for other in customer_claims
            if other.id != claim.id
        ),
    )
