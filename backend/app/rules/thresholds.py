"""Numeric thresholds of the deterministic core.

Changing a value here changes real decisions, so treat an edit as a policy change: update the tests that pin
the boundary and note the reason in the commit message.
"""

from decimal import Decimal

# --- routing -------------------------------------------------------------------------------------------
# Below this, the decision agent (or retrieval) is unsure enough that a human should decide instead.
MIN_DECISION_CONFIDENCE = Decimal("0.65")
# Above this, a claim always gets human review however clean it looks; it caps the cost of any single AI error.
AUTO_DECISION_MAX_AMOUNT = Decimal("100000.00")
# A risk score at or above this sends the claim to a human.
RISK_ESCALATION_THRESHOLD = 70
MAX_RISK_SCORE = 100

# --- rule parameters -----------------------------------------------------------------------------------
# R03: claims this soon after cover starts are a classic sign of insuring a loss that already happened.
NEW_POLICY_WINDOW_DAYS = 30
# R04: rolling window and the most claims a customer may make inside it before it counts as unusual.
CLAIM_FREQUENCY_WINDOW_DAYS = 365
MAX_CLAIMS_IN_WINDOW = 3

# --- risk weights --------------------------------------------------------------------------------------
# Serious signals equal the escalation threshold, so any one of them alone sends the claim to a human.
WEIGHT_AMOUNT_EXCEEDS_SUM_INSURED = RISK_ESCALATION_THRESHOLD  # payout must be capped by a person
WEIGHT_DUPLICATE_CLAIM = RISK_ESCALATION_THRESHOLD  # risk of paying the same loss twice
WEIGHT_KYC_INCOMPLETE = RISK_ESCALATION_THRESHOLD  # identity must be verified before any payment (AML)
# Weak signals that many honest claims also show: neither escalates alone, but both together (75) do.
WEIGHT_NEW_POLICY_CLAIM = 40
WEIGHT_FREQUENT_CLAIMANT = 35
