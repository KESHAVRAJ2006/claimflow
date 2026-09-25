"""Triage evaluation: labelled claims, per-case scoring and the summary report.

Pure functions over data: no database, no LLM. ``scripts.eval_triage`` runs the real pipeline and stores one
CaseResult per case; everything here turns stored results into metrics, so the report can be rebuilt at any time
without spending a token.

What is measured, and why:
- Unsafe approvals: APPROVE where the label says reject or escalate. The error the whole design exists to prevent;
  the target is zero.
- Accuracy: the outcome equals the label, or is one of its acceptable alternatives (escalating a claim that could
  have been approved costs a reviewer's time, not money).
- Coverage and citations: whether the decision agent read the policy right, and cited the clause that decides it.
- Escalations and retries: how much work reaches a senior reviewer, and how many reflection rounds bought nothing.
- Latency and tokens: what one claim costs, and so what the free tiers allow.
"""

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.state import EvidencePassage, ToolCallRecord
from app.domain.enums import IncidentType, ProductType, RecommendedOutcome
from app.rules.thresholds import AUTO_DECISION_MAX_AMOUNT

# Text the pipeline writes when a model call failed (llm.invoke_structured, the investigator's stop reason). A run
# that contains it measured the provider's quota, not the pipeline, so it is kept out of the accuracy figures.
PROVIDER_FAILURE_MARKERS = ("LLM call failed", "language model was unavailable")
# From llm_errors.explain_llm_error: one provider out of its daily quota, or each provider's limit when all failed.
_SINGLE_DAILY = "free daily quota is used up"
_DAILY, _PER_MINUTE = "(daily limit)", "(per-minute limit)"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---- the labelled set ---------------------------------------------------------------------------------------------


class ClaimSelector(_Model):
    """Which seeded claim supplies a case's facts: policy, customer, dates and amount.

    Either a seeded rule edge case, or the first claim (by number) matching the filters on which no rule triggers,
    so that the case's story alone decides the outcome.
    """

    edge_case: int | None = Field(default=None, ge=1, le=8, description="Seeded edge case CLM-<year>-9000NN (R01-R08)")
    product_type: ProductType | None = None
    incident_type: IncidentType | None = None
    # Below the auto-decision limit, so the amount alone never escalates a coverage case.
    max_amount: Decimal = AUTO_DECISION_MAX_AMOUNT
    min_policy_age_days: int | None = Field(default=None, ge=0, description="Incident date minus policy start")
    max_policy_age_days: int | None = Field(default=None, ge=0)
    first_policy: bool = Field(
        default=False, description="The customer held no earlier policy (no waiting period served)"
    )

    @model_validator(mode="after")
    def _one_way_to_select(self) -> "ClaimSelector":
        if (self.edge_case is None) == (self.product_type is None or self.incident_type is None):
            raise ValueError("give either edge_case, or product_type and incident_type")
        return self


class ExpectedCitation(_Model):
    """The clause that decides the case."""

    document: str
    section: str = Field(description='Clause number, e.g. "4.4"')

    def matches(self, document: str, section: str | None) -> bool:
        """Whether a cited passage is this clause.

        Args:
            document: Cited document.
            section: The passage's section heading, e.g. "4.4 Theft from an unlocked or unattended vehicle".

        Returns:
            True for the same document and clause number.
        """
        return document == self.document and bool(section) and section.split(" ", 1)[0] == self.section


class EvalCase(_Model):
    """One labelled claim."""

    id: str
    title: str
    kind: Literal["coverage", "rule"] = Field(description="coverage: the policy wording decides; rule: a rule does")
    base: ClaimSelector
    description: str = Field(min_length=10, description="What the claimant wrote; replaces the stored description")
    expected_outcome: RecommendedOutcome
    also_acceptable: tuple[RecommendedOutcome, ...] = ()
    expected_covered: bool | None = None
    expected_citation: ExpectedCitation | None = None
    why: str = Field(min_length=10, description="The reasoning behind the label, with the clause")

    @property
    def acceptable(self) -> frozenset[RecommendedOutcome]:
        """The expected outcome and every defensible alternative."""
        return frozenset((self.expected_outcome, *self.also_acceptable))


class EvalSet(_Model):
    """The whole labelled set."""

    cases: tuple[EvalCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalSet":
        duplicates = [case_id for case_id, count in Counter(c.id for c in self.cases).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        return self


def load_eval_set(path: Path) -> EvalSet:
    """Read and validate the labelled set.

    Args:
        path: JSON file.

    Returns:
        The cases.
    """
    return EvalSet.model_validate_json(path.read_text(encoding="utf-8"))


# ---- one run ------------------------------------------------------------------------------------------------------


class CitedSection(_Model):
    """A passage the recommendation cited."""

    document: str
    page: int
    section: str | None


class Usage(_Model):
    """LLM usage for one claim."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_by_model: dict[str, int] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


class CaseResult(_Model):
    """The pipeline's result for one case, stored one per line in the results file."""

    case_id: str
    claim_number: str = Field(description="The seeded claim that supplied the facts")
    ran_at: datetime
    status: Literal["completed", "failed", "provider_limited"]
    outcome: RecommendedOutcome | None = None
    covered: bool | None = None
    confidence: Decimal | None = None
    reasons: tuple[str, ...] = ()
    citations: tuple[CitedSection, ...] = ()
    investigation_rounds: int = 0
    retries: int = 0
    wasted_retries: int = 0
    open_questions: tuple[str, ...] = ()
    tool_calls: int = 0
    latency_s: float = 0.0
    usage: Usage = Usage()
    note: str | None = Field(default=None, description="Why a run failed or was limited")

    @property
    def scored(self) -> bool:
        """Whether this run counts in the metrics (a provider outage says nothing about the pipeline)."""
        return self.status != "provider_limited"


def count_wasted_retries(tool_calls: Sequence[ToolCallRecord], evidence: Sequence[EvidencePassage], rounds: int) -> int:
    """Count reflection retries that bought nothing.

    A retry is wasted when its investigation round neither made a tool call that no earlier round had made (same
    tool, same arguments) nor found a policy passage.

    Args:
        tool_calls: Every tool call of the run.
        evidence: Every passage retrieved.
        rounds: Investigation rounds run (1 = no retry).

    Returns:
        Wasted retries, between 0 and rounds - 1.
    """
    seen: set[tuple[str, str]] = set()
    wasted = 0
    for round_number in range(1, rounds + 1):
        calls = [call for call in tool_calls if call.round == round_number]
        keys = {(call.tool, json.dumps(call.args, sort_keys=True, default=str)) for call in calls}
        call_ids = {call.call_id for call in calls}
        found = [passage for passage in evidence if passage.found_by in call_ids]
        if round_number > 1 and not (keys - seen) and not found:
            wasted += 1
        seen |= keys
    return wasted


def provider_failure(texts: Iterable[str]) -> str | None:
    """Find a model-call failure in a run's reasons, flags and feedback.

    Args:
        texts: Everything the run wrote that could mention a failure.

    Returns:
        The first message naming a failed model call, or None.
    """
    return next((text for text in texts if any(marker in text for marker in PROVIDER_FAILURE_MARKERS)), None)


def is_daily_quota(message: str) -> bool:
    """Whether a provider failure will last until the quota resets tomorrow.

    Args:
        message: A provider failure message.

    Returns:
        True when the only provider, or every provider, is out of its daily quota. One provider on a per-minute
        limit recovers within a minute, so the evaluation carries on.
    """
    return _SINGLE_DAILY in message or (_DAILY in message and _PER_MINUTE not in message)


def load_results(path: Path) -> dict[str, CaseResult]:
    """Read the results file, keeping the latest run of each case.

    Args:
        path: JSON Lines file; missing means no results yet.

    Returns:
        Results by case id.
    """
    if not path.exists():
        return {}
    results: dict[str, CaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result = CaseResult.model_validate_json(line)
            results[result.case_id] = result
    return results


def append_result(path: Path, result: CaseResult) -> None:
    """Add one run to the results file, so an interrupted evaluation keeps everything it finished.

    Args:
        path: JSON Lines file.
        result: The run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(result.model_dump_json() + "\n")


# ---- scoring ------------------------------------------------------------------------------------------------------


class CaseScore(_Model):
    """How one result compares with its label."""

    case: EvalCase
    result: CaseResult
    exact: bool
    acceptable: bool
    unsafe_approval: bool = Field(description="Approved although the label says reject or escalate")
    wrongful_rejection: bool = Field(description="Rejected although the label says approve or escalate")
    unnecessary_escalation: bool
    covered_correct: bool | None
    citation_hit: bool | None


def score_case(case: EvalCase, result: CaseResult) -> CaseScore:
    """Compare one result with its label.

    Args:
        case: The labelled case.
        result: The pipeline's result.

    Returns:
        The score. A run with no outcome (intake failed) is neither exact nor acceptable.
    """
    outcome = result.outcome
    return CaseScore(
        case=case,
        result=result,
        exact=outcome is case.expected_outcome,
        acceptable=outcome in case.acceptable,
        unsafe_approval=outcome is RecommendedOutcome.APPROVE and outcome not in case.acceptable,
        wrongful_rejection=outcome is RecommendedOutcome.REJECT and outcome not in case.acceptable,
        unnecessary_escalation=outcome is RecommendedOutcome.ESCALATE
        and case.expected_outcome is not RecommendedOutcome.ESCALATE,
        covered_correct=None if case.expected_covered is None else result.covered is case.expected_covered,
        citation_hit=None
        if case.expected_citation is None
        else any(case.expected_citation.matches(c.document, c.section) for c in result.citations),
    )


class Summary(_Model):
    """Aggregate metrics over the scored runs."""

    total_cases: int
    scored: int
    missing: tuple[str, ...] = Field(description="Cases with no scored run yet")
    provider_limited: tuple[str, ...]
    exact: int
    acceptable: int
    unsafe_approvals: tuple[str, ...]
    wrongful_rejections: tuple[str, ...]
    escalated: int
    unnecessary_escalations: int
    covered_correct: int
    covered_labelled: int
    citation_hits: int
    citation_labelled: int
    claims_with_retries: int
    retries: int
    wasted_retries: int
    claims_with_open_questions: int
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_max_s: float | None
    mean_tokens: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_llm_calls: float | None
    by_kind: dict[str, tuple[int, int]] = Field(description="kind -> (acceptable, scored)")
    confusion: dict[str, dict[str, int]] = Field(description="expected -> outcome -> count")


def _percentile(values: Sequence[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest rank: the smallest value with at least `share` of the runs at or below it.
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def summarise(cases: Sequence[EvalCase], results: dict[str, CaseResult]) -> tuple[Summary, list[CaseScore]]:
    """Score every case that has a scored run.

    Args:
        cases: The labelled cases.
        results: Latest result per case id.

    Returns:
        The summary and the per-case scores, in case order.
    """
    scores = [score_case(case, results[case.id]) for case in cases if case.id in results and results[case.id].scored]
    runs = [score.result for score in scores]
    latencies = [run.latency_s for run in runs]
    by_kind: dict[str, tuple[int, int]] = {}
    for kind in ("coverage", "rule"):
        of_kind = [score for score in scores if score.case.kind == kind]
        by_kind[kind] = (sum(score.acceptable for score in of_kind), len(of_kind))
    confusion: dict[str, dict[str, int]] = {}
    for score in scores:
        row = confusion.setdefault(score.case.expected_outcome.value, {})
        got = score.result.outcome.value if score.result.outcome else "no outcome"
        row[got] = row.get(got, 0) + 1

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    summary = Summary(
        total_cases=len(cases),
        scored=len(scores),
        missing=tuple(case.id for case in cases if case.id not in results or not results[case.id].scored),
        provider_limited=tuple(case.id for case in cases if case.id in results and not results[case.id].scored),
        exact=sum(score.exact for score in scores),
        acceptable=sum(score.acceptable for score in scores),
        unsafe_approvals=tuple(score.case.id for score in scores if score.unsafe_approval),
        wrongful_rejections=tuple(score.case.id for score in scores if score.wrongful_rejection),
        escalated=sum(run.outcome is RecommendedOutcome.ESCALATE for run in runs),
        unnecessary_escalations=sum(score.unnecessary_escalation for score in scores),
        covered_correct=sum(bool(score.covered_correct) for score in scores),
        covered_labelled=sum(score.covered_correct is not None for score in scores),
        citation_hits=sum(bool(score.citation_hit) for score in scores),
        citation_labelled=sum(score.citation_hit is not None for score in scores),
        claims_with_retries=sum(run.retries > 0 for run in runs),
        retries=sum(run.retries for run in runs),
        wasted_retries=sum(run.wasted_retries for run in runs),
        claims_with_open_questions=sum(bool(run.open_questions) for run in runs),
        latency_p50_s=statistics.median(latencies) if latencies else None,
        latency_p95_s=_percentile(latencies, 0.95),
        latency_max_s=max(latencies) if latencies else None,
        mean_tokens=mean([float(run.usage.total_tokens) for run in runs]),
        mean_input_tokens=mean([float(run.usage.input_tokens) for run in runs]),
        mean_output_tokens=mean([float(run.usage.output_tokens) for run in runs]),
        mean_llm_calls=mean([float(run.usage.llm_calls) for run in runs]),
        by_kind=by_kind,
        confusion=confusion,
    )
    return summary, scores


# ---- report -------------------------------------------------------------------------------------------------------


class FreeTier(_Model):
    """A provider's free-tier limits, for the capacity estimate."""

    name: str
    tokens_per_minute: int
    requests_per_day: int


def _ratio(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "n/a"


def _mark(value: bool | None) -> str:
    return "–" if value is None else ("yes" if value else "**no**")


def render_report(summary: Summary, scores: Sequence[CaseScore], *, models: str, free_tier: FreeTier) -> str:
    """Format the results as Markdown.

    Args:
        summary: Aggregate metrics.
        scores: Per-case scores.
        models: Which models answered, for the header.
        free_tier: Limits of the primary provider's free tier.

    Returns:
        The report.
    """
    s = summary
    ran = max((score.result.ran_at for score in scores), default=None)
    lines = [
        "# Triage evaluation results",
        "",
        f"Generated by `python -m scripts.eval_triage --report`. Models: {models}. "
        f"Latest run: {ran:%Y-%m-%d %H:%M} UTC."
        if ran
        else "No runs yet.",
        "",
        f"Scored {s.scored} of {s.total_cases} cases.",
    ]
    if s.missing:
        lines.append(f"Not yet scored: {', '.join(s.missing)}.")
    if s.provider_limited:
        lines.append(f"Hit a provider limit (excluded; re-run to score): {', '.join(s.provider_limited)}.")
    tokens = f"{s.mean_tokens:,.0f}" if s.mean_tokens is not None else "n/a"
    calls = f"{s.mean_llm_calls:.1f}" if s.mean_llm_calls is not None else "n/a"
    lines += [
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        f"| Unsafe approvals (approved what should be rejected or escalated) | **{len(s.unsafe_approvals)}**"
        f"{' (' + ', '.join(s.unsafe_approvals) + ')' if s.unsafe_approvals else ''} | 0 |",
        f"| Outcome matches the label exactly | {_ratio(s.exact, s.scored)} | |",
        f"| Outcome acceptable (label or a defensible alternative) | {_ratio(s.acceptable, s.scored)} | |",
        f"| Rejections the label does not allow | {len(s.wrongful_rejections)}"
        f"{' (' + ', '.join(s.wrongful_rejections) + ')' if s.wrongful_rejections else ''} | 0 |",
        f"| Coverage judged correctly | {_ratio(s.covered_correct, s.covered_labelled)} | |",
        f"| Decisive clause cited | {_ratio(s.citation_hits, s.citation_labelled)} | |",
        f"| Escalation rate | {_ratio(s.escalated, s.scored)}, of which {s.unnecessary_escalations} "
        "not required by the label | |",
        f"| Reflection retries | {s.retries} in {s.claims_with_retries} claim(s), {s.wasted_retries} wasted "
        "| 0 wasted |",
        f"| Claims with questions left for the reviewer | {s.claims_with_open_questions} | |",
    ]
    if s.latency_p50_s is not None:
        lines.append(
            f"| Latency per claim | median {s.latency_p50_s:.0f} s, p95 {s.latency_p95_s:.0f} s, "
            f"max {s.latency_max_s:.0f} s | |"
        )
        lines.append(
            f"| LLM use per claim | {calls} calls, {tokens} tokens "
            f"({s.mean_input_tokens:,.0f} in / {s.mean_output_tokens:,.0f} out) | |"
        )
    for kind, (good, total) in s.by_kind.items():
        lines.append(f"| Acceptable outcomes, {kind} cases | {_ratio(good, total)} | |")

    if s.mean_tokens and s.mean_llm_calls:
        per_minute = free_tier.tokens_per_minute / s.mean_tokens
        per_day = free_tier.requests_per_day / s.mean_llm_calls
        lines += [
            "",
            f"**Free-tier capacity ({free_tier.name}).** At {tokens} tokens and {calls} calls per claim, the "
            f"{free_tier.tokens_per_minute:,} tokens-per-minute limit allows about {per_minute:.1f} claims a "
            f"minute, and {free_tier.requests_per_day:,} requests a day about {per_day:.0f} claims a day. Any "
            "token-per-day limit on the key applies on top; the provider console shows it.",
        ]

    lines += ["", "## Per case", "", "| Case | Expected | Got | OK | Covered right | Clause cited | Rounds | "
              "Questions for reviewer | Time | Tokens |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for score in scores:
        r, case = score.result, score.case
        got = r.outcome.value if r.outcome else f"none ({r.status})"
        flag = (
            "UNSAFE" if score.unsafe_approval else ("yes" if score.exact else ("ok" if score.acceptable else "**no**"))
        )
        lines.append(
            f"| {case.id}: {case.title} | {case.expected_outcome.value} | {got} | {flag} | "
            f"{_mark(score.covered_correct)} | {_mark(score.citation_hit)} | {r.investigation_rounds} | "
            f"{len(r.open_questions)} | {r.latency_s:.0f} s | {r.usage.total_tokens:,} |"
        )

    lines += ["", "## Confusion matrix (rows: label, columns: outcome)", ""]
    columns = ["approve", "reject", "escalate", "no outcome"]
    lines += ["| Label | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for expected in ("approve", "reject", "escalate"):
        row = s.confusion.get(expected, {})
        lines.append(f"| {expected} | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |")

    questions = [(score.case.id, question) for score in scores for question in score.result.open_questions]
    if questions:
        lines += ["", "## Questions the reflection agent passed to the reviewer", ""]
        lines += [f"- {case_id}: {question}" for case_id, question in questions]
    return "\n".join(lines) + "\n"
