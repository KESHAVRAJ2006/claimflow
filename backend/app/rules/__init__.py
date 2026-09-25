"""Deterministic core: fraud/eligibility rules, risk scoring and final routing.

DETERMINISTIC BY DESIGN. Nothing in this package calls an LLM, touches the database, reads the clock or does
any I/O. Agency is a cost paid for flexibility; these checks are known in advance, must give the same answer
every time, and must be explainable to an auditor or a court — so they are plain code with unit tests.
Agents may read a RiskReport, but no agent output can change one or override the routing it produces.
"""

from app.rules.engine import evaluate_all
from app.rules.models import (
    ClaimFacts,
    CustomerFacts,
    PolicyFacts,
    PriorClaim,
    RiskReport,
    RuleContext,
    RuleResult,
    Severity,
)
from app.rules.routing import RoutingDecision, RoutingInput, route_final

__all__ = [
    "ClaimFacts",
    "CustomerFacts",
    "PolicyFacts",
    "PriorClaim",
    "RiskReport",
    "RoutingDecision",
    "RoutingInput",
    "RuleContext",
    "RuleResult",
    "Severity",
    "evaluate_all",
    "route_final",
]
