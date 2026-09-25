"""Run every rule against a claim."""

from app.rules.checks import RULES
from app.rules.models import RiskReport, RuleContext


def evaluate_all(context: RuleContext) -> RiskReport:
    """Evaluate all registered rules.

    Every rule runs on every claim (no short-circuiting), so the report always shows the full picture:
    a reviewer sees that KYC passed as clearly as they see that the policy had lapsed.

    Args:
        context: Facts about the claim, its policy, the customer and their other claims.

    Returns:
        A RiskReport with one result per rule, in rule-ID order.
    """
    return RiskReport(results=tuple(registered.evaluate(context) for registered in RULES))
