"""Deterministic seed data: 50 customers, 80 policies, 200 claims, with labelled rule edge cases.

Usage (from backend/, or inside the container):
    python -m scripts.seed            # seeds or refreshes only when that cannot destroy app-written data
    python -m scripts.seed --reset    # wipes every table, then seeds

Dates are generated relative to the day you run it, so "claim 12 days after policy start" is only true on
that day. Without --reset the script therefore decides for itself (see ``decide_seed_action``):
empty database -> seed; untouched seed data from an earlier day -> refresh; seed data from today -> nothing;
data the app has written since seeding, or data this script did not create -> leave it alone.
docker compose runs it on every backend start. With the same ``--seed`` on the same day, output is identical.

Edge-case design: each labelled claim triggers exactly ONE rule and every unlabelled claim triggers NONE.
That keeps later evaluation unambiguous — if a clean claim gets flagged, the rule is wrong, not the data.
"""

import argparse
import asyncio
import random
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import AuditLog, Claim, ClaimRun, Customer, Policy, PremiumPayment
from app.db.session import create_db_engine, create_session_factory
from app.domain.enums import (
    ClaimStatus,
    FinalOutcome,
    IncidentType,
    KycStatus,
    PaymentStatus,
    PolicyStatus,
    ProductType,
)
from app.domain.products import INCIDENT_TYPES_BY_PRODUCT, POLICY_DOCUMENTS

NUM_CUSTOMERS = 50
NUM_POLICIES = 80
NUM_CLAIMS = 200
NUM_EDGE_CASE_CUSTOMERS = 8  # one dedicated customer per rule, so edge cases cannot interfere with each other

POLICY_TERM_DAYS = 365
INSTALLMENTS_PER_YEAR = 4
INSTALLMENT_INTERVAL_DAYS = 91  # quarterly billing; 4 x 91 = 364 keeps every installment inside the term
GRACE_PERIOD_DAYS = 30  # a missed installment lapses the policy this many days after its due date
RECENT_CLAIM_DAYS = 21  # claims with incidents this recent are still "submitted"; older ones were decided
# The newest regular claims form an untriaged backlog, so the queue has clean claims for the agent graph.
PENDING_REGULAR_CLAIMS = 15
# Up to ~7 years of policy history. R04 caps each customer at 3 claims per rolling year, so 42 regular
# customers need years of cover between them to hold 188 clean claims. 2600 was chosen with
# `python -m scripts.seed_capacity`: every seed from 1 to 200 builds, with at least 193 slots for 188 claims.
OLDEST_POLICY_START_DAYS = 2600
NEWEST_POLICY_START_DAYS = 60

# Threshold mirrored from the Phase 3 rules, used here only to keep unlabelled claims clean.
R03_NEW_POLICY_DAYS = 30
# Consecutive claims by one customer are at least 122 days apart, so any 4 of them span 366+ days and no
# 365-day window can hold more than 3 (R04). This is a guarantee by construction, not a retry loop.
MIN_CLAIM_GAP_DAYS = 122
CLAIM_GAP_JITTER_DAYS = 7  # extra random spacing so claim dates don't look machine-generated
FIRST_CLAIM_OFFSET_MAX_DAYS = 14  # how late after the first clean day a customer's first claim may fall

CENTS = Decimal("0.01")

FIRST_NAMES = (
    "Aarav", "Aditi", "Arjun", "Ananya", "Divya", "Farhan", "Gaurav", "Ishita", "Karan", "Kavya",
    "Meera", "Nikhil", "Neha", "Pooja", "Rahul", "Riya", "Rohan", "Sanjana", "Siddharth", "Sneha",
    "Tanvi", "Varun", "Vikram", "Zoya", "Harpreet", "Lakshmi", "Mohammed", "Joseph", "Anjali", "Deepak",
)  # fmt: skip
LAST_NAMES = (
    "Sharma", "Iyer", "Reddy", "Nair", "Gupta", "Khan", "Patel", "Menon", "Singh", "Das",
    "Joshi", "Rao", "Bose", "Kulkarni", "Fernandes", "Chopra", "Pillai", "Mehta", "Ghosh", "Verma",
)  # fmt: skip
CITIES = (
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Lucknow", "Chandigarh",
)  # fmt: skip
REVIEWERS = ("reviewer:priya.nair", "reviewer:amit.shah", "reviewer:fatima.sheikh", "reviewer:rajesh.kumar")

# Keyed by product too: natural calamity is both a motor and a home incident, and a single list once gave a motor
# claim "Roof and walls damaged by a falling tree". Every key keeps the same number of options as before, so the
# random choices, and with them every other seeded value, are unchanged.
DESCRIPTIONS: dict[tuple[ProductType, IncidentType], tuple[str, ...]] = {
    (ProductType.MOTOR, IncidentType.COLLISION): (
        "Rear-ended at a traffic signal; bumper and tail lamp damaged.",
        "Side collision while reversing out of a parking bay.",
    ),
    (ProductType.MOTOR, IncidentType.THEFT): ("Vehicle stolen from outside residence overnight; FIR filed.",),
    (ProductType.MOTOR, IncidentType.VANDALISM): ("Windscreen smashed and side panels scratched while parked.",),
    (ProductType.MOTOR, IncidentType.NATURAL_CALAMITY): (
        "Water entered the engine and interiors during heavy monsoon flooding.",
        "Roof and bonnet dented by a falling tree during a cyclone while parked.",
    ),
    (ProductType.HOME, IncidentType.NATURAL_CALAMITY): (
        "Floodwater entered the ground floor during heavy monsoon rain.",
        "Roof and walls damaged by a falling tree during a cyclone.",
    ),
    (ProductType.HEALTH, IncidentType.HOSPITALISATION): (
        "Admitted for four days with dengue fever.",
        "Admitted after a fall at home.",
    ),
    (ProductType.HEALTH, IncidentType.SURGERY): ("Planned knee arthroscopy.", "Emergency appendectomy."),
    (ProductType.HEALTH, IncidentType.DAY_CARE): ("Cataract procedure performed as day care.",),
    (ProductType.HOME, IncidentType.FIRE): ("Kitchen fire damaged cabinets and appliances.",),
    (ProductType.HOME, IncidentType.BURGLARY): (
        "Break-in while the family was travelling; electronics and jewellery taken.",
    ),
    (ProductType.HOME, IncidentType.WATER_DAMAGE): (
        "Burst pipe in the upstairs bathroom damaged ceilings and flooring.",
    ),
}


@dataclass(frozen=True)
class ProductProfile:
    """Generation parameters for one product line."""

    prefix: str  # policy number prefix
    sums_insured: tuple[int, ...]
    deductible: int
    premium_rate: Decimal  # annual premium as a fraction of sum insured
    claim_range: tuple[int, int]  # typical claim amount bounds
    weight: int  # relative share of the policy book


PRODUCTS: dict[ProductType, ProductProfile] = {
    ProductType.MOTOR: ProductProfile(
        "MOT", (300_000, 500_000, 750_000, 1_000_000, 1_500_000), 5_000, Decimal("0.025"), (8_000, 180_000), 45
    ),
    ProductType.HEALTH: ProductProfile(
        "HLT", (300_000, 500_000, 1_000_000), 10_000, Decimal("0.035"), (15_000, 350_000), 35
    ),
    ProductType.HOME: ProductProfile(
        "HOM", (1_000_000, 2_000_000, 3_500_000, 5_000_000), 10_000, Decimal("0.003"), (20_000, 400_000), 20
    ),
}


@dataclass(frozen=True)
class EdgeCase:
    """A claim deliberately built to trigger exactly one deterministic rule."""

    claim_number: str
    rule_id: str
    rule_name: str
    scenario: str


@dataclass
class SeedData:
    """Everything the seed writes, in foreign-key order."""

    customers: list[Customer] = field(default_factory=list)
    policies: list[Policy] = field(default_factory=list)
    payments: list[PremiumPayment] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    audit_entries: list[AuditLog] = field(default_factory=list)
    edge_cases: list[EdgeCase] = field(default_factory=list)


class _DatasetBuilder:
    """Stateful helper that generates rows with a single seeded random generator."""

    def __init__(self, now: datetime, seed: int) -> None:
        """Prepare an empty dataset.

        Args:
            now: The moment the seed runs; all generated timestamps are at or before it.
            seed: Random seed for reproducible output.
        """
        self.now = now
        self.today = now.date()
        self.rng = random.Random(seed)  # noqa: S311 — reproducible test data, not cryptography
        self.data = SeedData()
        self._policy_seq = 0
        self._claim_seq = 0
        self._edge_seq = 0

    # ---- small helpers -------------------------------------------------------------------------

    def _uuid(self) -> uuid.UUID:
        """Return a UUID drawn from the seeded generator, so IDs are reproducible too."""
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)

    def _days_ago(self, days: int) -> date:
        """Return the date ``days`` before today (negative values are in the future)."""
        return self.today - timedelta(days=days)

    def _date_between(self, low: date, high: date) -> date:
        """Return a uniformly random date in the inclusive range [low, high]."""
        return low + timedelta(days=self.rng.randint(0, (high - low).days))

    def _timestamp_on(self, day: date) -> datetime:
        """Return a UTC timestamp during business hours on ``day``, never later than now."""
        moment = datetime.combine(day, time(self.rng.randint(8, 19), self.rng.randint(0, 59)), tzinfo=UTC)
        return min(moment, self.now)

    def _claim_amount(self, policy: Policy) -> Decimal:
        """Draw a claim amount for a policy, rounded to 500 and never above the sum insured."""
        low, high = PRODUCTS[policy.product_type].claim_range
        # Squaring a uniform draw skews toward small claims, which dominate real claim books.
        raw = low + (high - low) * self.rng.random() ** 2
        amount = Decimal(int(raw // 500 * 500))
        return min(amount, policy.sum_insured).quantize(CENTS)

    # ---- row factories ------------------------------------------------------------------------

    def add_customer(self, index: int, kyc_status: KycStatus = KycStatus.VERIFIED) -> Customer:
        """Create a customer.

        Args:
            index: 1-based sequence number used in the reference and email.
            kyc_status: Identity verification state.

        Returns:
            The new Customer (also appended to the dataset).
        """
        first, last = self.rng.choice(FIRST_NAMES), self.rng.choice(LAST_NAMES)
        # Onboarded before the oldest possible policy start, so no policy predates its customer.
        onboarded_days_ago = self.rng.randint(OLDEST_POLICY_START_DAYS + 20, OLDEST_POLICY_START_DAYS + 400)
        created = self._timestamp_on(self._days_ago(onboarded_days_ago))
        customer = Customer(
            id=self._uuid(),
            customer_ref=f"CUS-{index:05d}",
            full_name=f"{first} {last}",
            # example.com is reserved for documentation (RFC 2606), so seeded emails can never reach a real inbox.
            email=f"{first}.{last}.{index}@example.com".lower(),
            phone=f"+91 9{self.rng.randint(100_000_000, 999_999_999)}",
            date_of_birth=self._days_ago(self.rng.randint(22 * 365, 70 * 365)),
            city=self.rng.choice(CITIES),
            kyc_status=kyc_status,
            created_at=created,
            updated_at=created,
        )
        self.data.customers.append(customer)
        return customer

    def add_policy(
        self,
        customer: Customer,
        product: ProductType,
        start_date: date,
        *,
        sum_insured: int | None = None,
        lapse_installment: int | None = None,
    ) -> Policy:
        """Create a one-year policy and its premium installments.

        Args:
            customer: Policyholder.
            product: Product line.
            start_date: First day of cover.
            sum_insured: Fixed sum insured; random for the product when None.
            lapse_installment: Zero-based installment index that goes unpaid, lapsing the policy. None = no lapse.

        Returns:
            The new Policy (also appended to the dataset).
        """
        profile = PRODUCTS[product]
        self._policy_seq += 1
        end_date = start_date + timedelta(days=POLICY_TERM_DAYS - 1)  # inclusive last day of cover
        insured = Decimal(sum_insured if sum_insured is not None else self.rng.choice(profile.sums_insured))
        premium = (insured * profile.premium_rate).quantize(Decimal("1")).quantize(CENTS)  # whole rupees

        lapsed_on = None
        if lapse_installment is not None:
            missed_due = start_date + timedelta(days=INSTALLMENT_INTERVAL_DAYS * lapse_installment)
            lapsed_on = missed_due + timedelta(days=GRACE_PERIOD_DAYS)

        if lapsed_on is not None:
            status = PolicyStatus.LAPSED
        elif end_date < self.today:
            status = PolicyStatus.EXPIRED
        else:
            status = PolicyStatus.ACTIVE

        created = self._timestamp_on(start_date - timedelta(days=self.rng.randint(1, 14)))
        policy = Policy(
            id=self._uuid(),
            policy_number=f"{profile.prefix}-{start_date.year}-{self._policy_seq:06d}",
            customer_id=customer.id,
            product_type=product,
            status=status,
            start_date=start_date,
            end_date=end_date,
            lapsed_on=lapsed_on,
            sum_insured=insured.quantize(CENTS),
            deductible=Decimal(profile.deductible).quantize(CENTS),
            annual_premium=premium,
            policy_document=POLICY_DOCUMENTS[product],
            created_at=created,
            updated_at=created,
        )
        self.data.policies.append(policy)
        self._add_payments(policy, lapse_installment)
        return policy

    def _add_payments(self, policy: Policy, lapse_installment: int | None) -> None:
        """Generate the quarterly premium installments that have fallen due so far.

        Args:
            policy: The policy being billed.
            lapse_installment: Zero-based index of the installment that is missed, if any.
        """
        base = (policy.annual_premium / INSTALLMENTS_PER_YEAR).quantize(CENTS, rounding=ROUND_DOWN)
        # The final installment absorbs the rounding remainder so the four always sum exactly to the premium.
        amounts = [base] * (INSTALLMENTS_PER_YEAR - 1) + [policy.annual_premium - base * (INSTALLMENTS_PER_YEAR - 1)]

        for index, amount in enumerate(amounts):
            due = policy.start_date + timedelta(days=INSTALLMENT_INTERVAL_DAYS * index)
            if due > self.today:
                break  # not billed yet
            if index == lapse_installment:
                status, paid = PaymentStatus.MISSED, None
            elif due <= self._days_ago(21) and self.rng.random() < 0.1:
                status, paid = PaymentStatus.LATE, due + timedelta(days=self.rng.randint(1, 20))
            else:
                status, paid = PaymentStatus.PAID, due - timedelta(days=self.rng.randint(0, 7))
            self.data.payments.append(
                PremiumPayment(
                    id=self._uuid(),
                    policy_id=policy.id,
                    installment_number=index + 1,
                    due_date=due,
                    paid_date=paid,
                    amount=amount,
                    status=status,
                    created_at=self._timestamp_on(due),
                )
            )
            if status is PaymentStatus.MISSED:
                break  # an insurer stops billing once a policy lapses

    def add_claim(
        self,
        policy: Policy,
        incident_date: date,
        amount: Decimal,
        *,
        status: ClaimStatus | None = None,
        created_on: date | None = None,
        claim_number: str | None = None,
    ) -> Claim:
        """Create a claim plus its audit entries.

        Args:
            policy: Policy the claim is made under.
            incident_date: When the loss happened.
            amount: Claimed amount.
            status: Fixed status; when None, recent incidents are "submitted" and older ones were decided.
            created_on: Submission date; defaults to 0-10 days after the incident.
            claim_number: Fixed claim number; generated when None.

        Returns:
            The new Claim (also appended to the dataset).
        """
        if claim_number is None:
            self._claim_seq += 1
            claim_number = f"CLM-{incident_date.year}-{self._claim_seq:06d}"
        submitted_on = created_on or incident_date + timedelta(days=self.rng.randint(0, 10))
        created = self._timestamp_on(submitted_on)

        if status is None:
            if incident_date >= self._days_ago(RECENT_CLAIM_DAYS):
                status = ClaimStatus.SUBMITTED
            else:
                # Roughly a quarter of historical claims were rejected, a plausible book-wide rate.
                status = ClaimStatus.APPROVED if self.rng.random() < 0.75 else ClaimStatus.REJECTED

        final_outcome = decided_by = decided_at = None
        if status in (ClaimStatus.APPROVED, ClaimStatus.REJECTED):
            final_outcome = FinalOutcome.APPROVE if status is ClaimStatus.APPROVED else FinalOutcome.REJECT
            decided_by = self.rng.choice(REVIEWERS)
            decided_at = min(created + timedelta(days=self.rng.randint(1, 10), hours=self.rng.randint(0, 6)), self.now)

        claim = Claim(
            id=self._uuid(),
            claim_number=claim_number,
            policy_id=policy.id,
            incident_type=self.rng.choice(INCIDENT_TYPES_BY_PRODUCT[policy.product_type]),
            incident_date=incident_date,
            description="",
            claimed_amount=amount,
            status=status,
            final_outcome=final_outcome,
            decided_by=decided_by,
            decided_at=decided_at,
            created_at=created,
            updated_at=decided_at or created,
        )
        claim.description = self.rng.choice(DESCRIPTIONS[(policy.product_type, claim.incident_type)])
        self.data.claims.append(claim)

        # JSON has no decimal type; money goes into JSONB as a string so no precision is lost.
        self.data.audit_entries.append(
            AuditLog(
                claim_id=claim.id,
                actor="customer_portal",
                action="claim.submitted",
                after={"claim_number": claim_number, "claimed_amount": str(amount), "status": "submitted"},
                created_at=created,
            )
        )
        if final_outcome is not None and decided_by is not None:
            self.data.audit_entries.append(
                AuditLog(
                    claim_id=claim.id,
                    actor=decided_by,
                    action="claim.decided",
                    before={"status": "submitted"},
                    after={"status": status.value, "final_outcome": final_outcome.value},
                    created_at=decided_at,
                )
            )
        return claim

    def _add_edge_claim(
        self, policy: Policy, incident_date: date, amount: str, rule_id: str, rule_name: str, scenario: str,
        *, created_on: date | None = None,
    ) -> None:  # fmt: skip
        """Create a submitted claim reserved for a rule edge case and record its label."""
        self._edge_seq += 1
        # The 9xxxxx range keeps edge-case claim numbers recognisable at a glance in the UI.
        number = f"CLM-{self.today.year}-9{self._edge_seq:05d}"
        self.add_claim(
            policy, incident_date, Decimal(amount), status=ClaimStatus.SUBMITTED, created_on=created_on,
            claim_number=number,
        )  # fmt: skip
        self.data.edge_cases.append(EdgeCase(number, rule_id, rule_name, scenario))

    # ---- dataset sections ---------------------------------------------------------------------

    def build_edge_cases(self, first_customer_index: int) -> None:
        """Create 8 isolated customers whose claims each trigger exactly one rule.

        Args:
            first_customer_index: Customer sequence number for the first edge-case customer.
        """
        indexes = iter(range(first_customer_index, first_customer_index + NUM_EDGE_CASE_CUSTOMERS))
        ago = self._days_ago

        def new_policy(product: ProductType, start: date, kyc: KycStatus = KycStatus.VERIFIED, **kwargs: int) -> Policy:
            return self.add_policy(self.add_customer(next(indexes), kyc), product, start, **kwargs)

        policy = new_policy(ProductType.MOTOR, ago(200), sum_insured=500_000)
        self._add_edge_claim(
            policy, ago(40), "650000.00", "R01", "amount_exceeds_sum_insured",
            "Claimed 650,000.00 against a 500,000.00 sum insured",
        )  # fmt: skip

        # Installment index 2 is due on day 182 and missed, so the policy lapses on day 212 (88 days ago).
        policy = new_policy(ProductType.HEALTH, ago(300), lapse_installment=2)
        self._add_edge_claim(
            policy, ago(50), "72000.00", "R02", "policy_lapsed_at_incident_date",
            f"Policy lapsed on {policy.lapsed_on}; incident on {ago(50)}",
        )  # fmt: skip

        policy = new_policy(ProductType.MOTOR, ago(20))
        incident = policy.start_date + timedelta(days=12)
        self._add_edge_claim(
            policy, incident, "45000.00", "R03", "claim_within_30d_of_policy_start",
            f"Incident {incident} is 12 days after policy start {policy.start_date}",
        )  # fmt: skip

        # Three already-approved claims followed by a fourth, all inside six months.
        policy = new_policy(ProductType.HOME, ago(250))
        for days, amount in ((170, "31500.00"), (120, "48000.00"), (75, "27500.00")):
            self.add_claim(policy, ago(days), Decimal(amount), status=ClaimStatus.APPROVED)
        self._add_edge_claim(
            policy, ago(30), "56000.00", "R04", "more_than_3_claims_in_12_months",
            "4th claim by the same customer in under 6 months",
        )  # fmt: skip

        policy = new_policy(ProductType.MOTOR, ago(150))
        self._add_edge_claim(
            policy, ago(-5), "25000.00", "R05", "incident_date_in_future",
            f"Submitted {self.today} for an incident dated {ago(-5)}",
            created_on=self.today,
        )  # fmt: skip

        policy = new_policy(ProductType.HEALTH, ago(60))
        self._add_edge_claim(
            policy, ago(75), "80000.00", "R06", "incident_date_before_policy_start",
            f"Incident {ago(75)} is 15 days before policy start {policy.start_date}",
            created_on=ago(2),
        )  # fmt: skip

        policy = new_policy(ProductType.MOTOR, ago(180))
        self.add_claim(policy, ago(25), Decimal("38500.00"), status=ClaimStatus.APPROVED, created_on=ago(24))
        self._add_edge_claim(
            policy, ago(25), "38500.00", "R07", "duplicate_claim_same_date_amount",
            "Same policy, incident date and amount as an earlier approved claim",
            created_on=ago(10),
        )  # fmt: skip

        policy = new_policy(ProductType.HEALTH, ago(200), kyc=KycStatus.PENDING)
        self._add_edge_claim(
            policy, ago(35), "42000.00", "R08", "kyc_incomplete", "Customer KYC status is 'pending'",
        )  # fmt: skip

    def build_regular_book(self, customer_count: int, policy_count: int, claim_count: int) -> None:
        """Create the unlabelled customers, policies and claims, none of which trigger any rule.

        Args:
            customer_count: Number of regular customers.
            policy_count: Number of regular policies (one or two per customer).
            claim_count: Number of regular claims.

        Raises:
            ValueError: If there are more than two policies per customer to distribute.
            RuntimeError: If the policies do not offer enough rule-clean claim dates.
        """
        extra_policies = policy_count - customer_count
        if not 0 <= extra_policies <= customer_count:
            raise ValueError("policy_count must be between customer_count and 2 * customer_count")
        customers = [self.add_customer(index) for index in range(1, customer_count + 1)]

        # Spread renewals evenly (a random share left too many single-policy customers, who can hold only
        # ~3 clean claims each, so 188 claims did not fit).
        policies_per_customer = Counter({customer.id: 1 for customer in customers})
        for customer in self.rng.sample(customers, extra_policies):
            policies_per_customer[customer.id] += 1

        candidates: list[tuple[date, Policy]] = []
        for customer in customers:
            chain = self._add_renewal_chain(customer, policies_per_customer[customer.id])
            candidates.extend(self._claim_slots(chain))
        if len(candidates) < claim_count:
            raise RuntimeError(f"Only {len(candidates)} clean claim slots for {claim_count} regular claims")

        # Dropping random slots only ever widens gaps, so every R04/R07 guarantee still holds after sampling.
        chosen = self.rng.sample(candidates, claim_count)
        # Creating claims in incident order makes claim numbers chronological, like a real system.
        ordered = sorted(chosen, key=lambda item: (item[0], item[1].policy_number))
        backlog_starts_at = len(ordered) - PENDING_REGULAR_CLAIMS
        for position, (incident, policy) in enumerate(ordered):
            status = ClaimStatus.SUBMITTED if position >= backlog_starts_at else None
            # The amount is capped at sum insured, so R01 never fires.
            self.add_claim(policy, incident, self._claim_amount(policy), status=status)

    def _add_renewal_chain(self, customer: Customer, terms: int) -> list[Policy]:
        """Create back-to-back yearly renewals of one product, oldest first.

        Args:
            customer: Policyholder.
            terms: Number of consecutive one-year policies.

        Returns:
            The policies in chronological order.
        """
        products = list(PRODUCTS)
        product = self.rng.choices(products, weights=[PRODUCTS[p].weight for p in products])[0]
        # The latest term starts late enough that the oldest renewal still falls inside the history horizon.
        latest_start = self._days_ago(
            self.rng.randint(NEWEST_POLICY_START_DAYS, OLDEST_POLICY_START_DAYS - POLICY_TERM_DAYS * (terms - 1))
        )
        chain = []
        for term in range(terms - 1, -1, -1):
            start = latest_start - timedelta(days=POLICY_TERM_DAYS * term)
            # Only the current term may lapse; a renewal issued after a lapse would be contradictory.
            lapse = self._choose_lapse(start) if term == 0 else None
            chain.append(self.add_policy(customer, product, start, lapse_installment=lapse))
        return chain

    def _choose_lapse(self, start: date) -> int | None:
        """Decide whether a policy lapses and at which installment.

        Args:
            start: Policy start date.

        Returns:
            Zero-based index of the missed installment, or None if the policy stays in force.
        """
        last_possible_lapse = min(start + timedelta(days=POLICY_TERM_DAYS - 1), self.today)
        options = [
            index
            for index in range(1, INSTALLMENTS_PER_YEAR)
            if start + timedelta(days=INSTALLMENT_INTERVAL_DAYS * index + GRACE_PERIOD_DAYS) <= last_possible_lapse
        ]
        # About 1 in 7 eligible policies lapse, enough for the payment-history tool to have something to find.
        return self.rng.choice(options) if options and self.rng.random() < 0.15 else None

    def _claim_slots(self, chain: list[Policy]) -> list[tuple[date, Policy]]:
        """List rule-clean incident dates across a customer's policies, spaced at least 122 days apart.

        Args:
            chain: The customer's policies in chronological order.

        Returns:
            (incident_date, policy) pairs that trigger none of R01-R08.
        """
        windows = []
        for policy in chain:
            # From day 31 (clear of R03) to the day before any lapse (clear of R02) and no later than
            # yesterday (clear of R05).
            first = policy.start_date + timedelta(days=R03_NEW_POLICY_DAYS + 1)
            last_covered = policy.lapsed_on - timedelta(days=1) if policy.lapsed_on else policy.end_date
            last = min(last_covered, self._days_ago(1))
            if last >= first:
                windows.append((first, last, policy))
        if not windows:
            return []

        slots = []
        cursor = windows[0][0] + timedelta(days=self.rng.randint(0, FIRST_CLAIM_OFFSET_MAX_DAYS))
        for first, last, policy in windows:
            cursor = max(cursor, first)  # skip the gap between one term's end and the next term's day 31
            while cursor <= last:
                slots.append((cursor, policy))
                # Jitter keeps dates irregular; the later random sampling makes the real gaps wider still.
                gap = MIN_CLAIM_GAP_DAYS + self.rng.randint(0, CLAIM_GAP_JITTER_DAYS)
                cursor += timedelta(days=gap)
        return slots


def build_dataset(now: datetime, seed: int = 42) -> SeedData:
    """Generate the full seed dataset in memory without touching the database.

    Args:
        now: Timezone-aware "current time" that all dates are relative to.
        seed: Random seed; the same seed and ``now`` give identical data.

    Returns:
        The generated rows and the labelled edge cases.

    Raises:
        ValueError: If ``now`` is naive.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    builder = _DatasetBuilder(now, seed)
    regular_customers = NUM_CUSTOMERS - NUM_EDGE_CASE_CUSTOMERS
    builder.build_edge_cases(first_customer_index=regular_customers + 1)
    edge_claims = len(builder.data.claims)
    builder.build_regular_book(
        customer_count=regular_customers,
        policy_count=NUM_POLICIES - NUM_EDGE_CASE_CUSTOMERS,
        claim_count=NUM_CLAIMS - edge_claims,
    )
    # Present customers in reference order regardless of which section created them first.
    builder.data.customers.sort(key=lambda customer: customer.customer_ref)
    return builder.data


async def write_dataset(session: AsyncSession, data: SeedData) -> None:
    """Insert the dataset inside the caller's transaction.

    Args:
        session: An open session with an active transaction.
        data: Rows produced by ``build_dataset``.
    """
    # Flushing one table at a time guarantees parents exist before children reference them.
    for batch in (data.customers, data.policies, data.payments, data.claims, data.audit_entries):
        session.add_all(batch)
        await session.flush()


def _print_summary(data: SeedData) -> None:
    """Print row counts and the labelled edge cases.

    Args:
        data: The dataset that was written.
    """
    print(f"Seeded {len(data.customers)} customers, {len(data.policies)} policies, "
          f"{len(data.payments)} premium payments, {len(data.claims)} claims, "
          f"{len(data.audit_entries)} audit entries.")  # fmt: skip
    status_counts = sorted(Counter(claim.status.value for claim in data.claims).items())
    print("Claims by status: " + ", ".join(f"{status}={count}" for status, count in status_counts))
    print("\nLabelled edge cases (each triggers exactly one rule):")
    print(f"  {'CLAIM':<17} {'RULE':<4} {'NAME':<34} SCENARIO")
    for case in data.edge_cases:
        print(f"  {case.claim_number:<17} {case.rule_id:<4} {case.rule_name:<34} {case.scenario}")


SEED_COMPLETED_ACTION = "seed.completed"


class SeedAction(StrEnum):
    """What ``run`` does, decided from what is already in the database."""

    SEED = "seed"  # empty database
    RESET = "reset"  # --reset was passed: wipe whatever is there
    REFRESH = "refresh"  # untouched seed data from an earlier day; its relative dates have drifted
    UP_TO_DATE = "up_to_date"  # untouched seed data from today
    KEEP_USED_DATA = "keep_used_data"  # the app has written to the data since seeding
    KEEP_UNKNOWN_DATA = "keep_unknown_data"  # data this script did not create


@dataclass(frozen=True)
class DatabaseState:
    """Facts about the database that decide whether seeding is safe."""

    has_data: bool
    seeded_on: date | None  # day of the latest completed seed; None if this script never seeded it
    used_since_seed: bool  # anything written after the seed finished (claim runs, decisions, new claims)


def decide_seed_action(state: DatabaseState, today: date, reset: bool) -> SeedAction:
    """Choose what seeding should do. Only an explicit --reset may delete data the app has written.

    Args:
        state: Current database facts.
        today: Today's date in UTC.
        reset: Whether --reset was passed.

    Returns:
        The action to take.
    """
    if reset:
        return SeedAction.RESET
    if not state.has_data:
        return SeedAction.SEED
    if state.seeded_on is None:
        return SeedAction.KEEP_UNKNOWN_DATA
    if state.used_since_seed:
        return SeedAction.KEEP_USED_DATA
    if state.seeded_on == today:
        return SeedAction.UP_TO_DATE
    return SeedAction.REFRESH


async def read_database_state(session: AsyncSession) -> DatabaseState:
    """Inspect the database to decide whether seeding is safe.

    Args:
        session: An open session.

    Returns:
        The current DatabaseState.
    """
    has_data = bool(await session.scalar(select(func.count()).select_from(Customer)))
    latest_seed = (
        await session.execute(
            select(AuditLog.id, AuditLog.after, AuditLog.created_at)
            .where(AuditLog.action == SEED_COMPLETED_ACTION)
            .order_by(AuditLog.id.desc())
            .limit(1)
        )
    ).first()
    if latest_seed is None or latest_seed.after is None:
        return DatabaseState(has_data=has_data, seeded_on=None, used_since_seed=False)

    newest_audit_id = await session.scalar(select(func.max(AuditLog.id)))
    has_claim_runs = bool(await session.scalar(select(func.count()).select_from(ClaimRun)))
    # Belt and braces: even a write that forgot its audit row still bumps updated_at.
    last_modified = [await session.scalar(select(func.max(model.updated_at))) for model in (Customer, Policy, Claim)]
    used_since_seed = (
        has_claim_runs
        or (newest_audit_id or 0) > latest_seed.id
        or any(ts is not None and ts > latest_seed.created_at for ts in last_modified)
    )
    return DatabaseState(
        has_data=has_data,
        seeded_on=date.fromisoformat(latest_seed.after["seeded_on"]),
        used_since_seed=used_since_seed,
    )


def describe_skip(action: SeedAction, state: DatabaseState, today: date) -> str:
    """Explain why seeding did nothing.

    Args:
        action: One of the actions that leave the database untouched.
        state: Current database facts.
        today: Today's date in UTC.

    Returns:
        A human-readable message.
    """
    reset_hint = "Run `python -m scripts.seed --reset` to wipe and reseed."
    if action is SeedAction.UP_TO_DATE:
        return "seed: today's seed data is already loaded and untouched; nothing to do."
    if action is SeedAction.KEEP_UNKNOWN_DATA:
        return f"seed: the database holds data this script did not create, so it was left untouched. {reset_hint}"
    drift = ""
    if state.seeded_on is not None and state.seeded_on != today:
        drift = (
            f" It was seeded on {state.seeded_on}, so date-relative edge cases (e.g. 'claim 12 days after "
            "policy start') have drifted."
        )
    return (
        "seed: the app has written data since seeding (claim runs, decisions or new claims), so it was left "
        f"untouched rather than deleting that work.{drift} {reset_hint}"
    )


async def run(reset: bool, seed: int) -> int:
    """Seed the configured database when it is safe to do so.

    Args:
        reset: Truncate all tables first, regardless of their contents.
        seed: Random seed.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    if settings.is_production:
        print("seed: refusing to run with ENVIRONMENT=production.", file=sys.stderr)
        return 1

    now = datetime.now(UTC)
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session, session.begin():
            state = await read_database_state(session)
            action = decide_seed_action(state, today=now.date(), reset=reset)
            if action in (SeedAction.UP_TO_DATE, SeedAction.KEEP_USED_DATA, SeedAction.KEEP_UNKNOWN_DATA):
                print(describe_skip(action, state, now.date()))
                return 0
            if action in (SeedAction.RESET, SeedAction.REFRESH):
                # TRUNCATE bypasses the audit_log UPDATE/DELETE trigger by design; only this dev script uses it.
                await session.execute(
                    text(
                        "TRUNCATE audit_log, claim_runs, claims, premium_payments, policies, customers RESTART IDENTITY"
                    )
                )
            data = build_dataset(now=now, seed=seed)
            await write_dataset(session, data)
            # Written last, so it has the highest audit id: any later audit row means the app used the data.
            session.add(
                AuditLog(
                    actor="seed",
                    action=SEED_COMPLETED_ACTION,
                    after={"seeded_on": now.date().isoformat(), "seed": seed, "claims": len(data.claims)},
                    created_at=now,
                )
            )
            await session.flush()
    finally:
        await engine.dispose()

    if action is SeedAction.REFRESH:
        print(f"seed: seed data from {state.seeded_on} was untouched but stale; reloaded it for today.")
    _print_summary(data)
    return 0


def main() -> None:
    """Parse CLI arguments and run the seed."""
    parser = argparse.ArgumentParser(description="Seed ClaimFlow with deterministic demo data.")
    parser.add_argument("--reset", action="store_true", help="wipe all tables before seeding, even app-written data")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(reset=args.reset, seed=args.seed)))


if __name__ == "__main__":
    main()
