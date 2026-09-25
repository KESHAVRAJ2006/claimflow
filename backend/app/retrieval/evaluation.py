"""Retrieval evaluation: Recall@k and MRR against page-level relevance labels."""

import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ProductType
from app.retrieval.retriever import PolicyRetriever


class RelevantPage(BaseModel):
    """A (document, page) pair that answers a question."""

    model_config = ConfigDict(frozen=True)

    document: str
    page: int = Field(ge=1)


class EvalQuestion(BaseModel):
    """A labelled question."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    product_type: ProductType
    relevant: tuple[RelevantPage, ...] = Field(min_length=1)


class EvalSet(BaseModel):
    """The full labelled set."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(default=5, ge=1)
    questions: tuple[EvalQuestion, ...] = Field(min_length=10)
    off_topic: tuple[str, ...] = Field(default=(), description="Unanswerable questions, for confidence calibration")


class QuestionResult(BaseModel):
    """Outcome for one question in one mode."""

    model_config = ConfigDict(frozen=True)

    id: str
    filtered: bool
    retrieved: tuple[RelevantPage, ...]
    recall: float
    reciprocal_rank: float
    top_score: float
    confidence: float


class EvalReport(BaseModel):
    """Aggregate metrics."""

    model_config = ConfigDict(frozen=True)

    k: int
    filtered_recall: float
    unfiltered_recall: float
    filtered_mrr: float
    unfiltered_mrr: float
    results: tuple[QuestionResult, ...]
    answerable_confidences: tuple[float, ...]
    answerable_top_scores: tuple[float, ...]
    off_topic_confidences: tuple[float, ...]
    off_topic_top_scores: tuple[float, ...]

    @property
    def median_answerable_confidence(self) -> float:
        """Median confidence for answerable questions (filtered mode)."""
        return statistics.median(self.answerable_confidences) if self.answerable_confidences else 0.0

    @property
    def max_off_topic_confidence(self) -> float:
        """Highest confidence given to an off-topic question."""
        return max(self.off_topic_confidences, default=0.0)


def load_eval_set(path: Path) -> EvalSet:
    """Load and validate a labelled question file.

    Args:
        path: JSON file path.

    Returns:
        The validated EvalSet.
    """
    return EvalSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def recall_at_k(retrieved: Sequence[RelevantPage], relevant: Sequence[RelevantPage]) -> float:
    """Fraction of relevant pages that appear among the retrieved pages.

    Args:
        retrieved: Pages of the top-k hits, in rank order (duplicates allowed).
        relevant: Labelled relevant pages.

    Returns:
        A value in [0, 1].
    """
    return len(set(relevant) & set(retrieved)) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[RelevantPage], relevant: Sequence[RelevantPage]) -> float:
    """1 / rank of the first relevant hit, or 0 if none is retrieved.

    Args:
        retrieved: Pages of the hits, in rank order.
        relevant: Labelled relevant pages.

    Returns:
        A value in [0, 1].
    """
    wanted = set(relevant)
    for rank, page in enumerate(retrieved, start=1):
        if page in wanted:
            return 1 / rank
    return 0.0


async def run_retrieval_eval(retriever: PolicyRetriever, eval_set: EvalSet) -> EvalReport:
    """Evaluate every question with and without the product filter.

    Args:
        retriever: A retriever whose collection already holds the policy documents.
        eval_set: Labelled questions.

    Returns:
        The aggregate report.
    """
    results: list[QuestionResult] = []
    for question in eval_set.questions:
        for filtered in (True, False):
            search = await retriever.search(
                question.question, product_type=question.product_type if filtered else None, limit=eval_set.k
            )
            retrieved = tuple(RelevantPage(document=c.citation.document, page=c.citation.page) for c in search.chunks)
            results.append(
                QuestionResult(
                    id=question.id,
                    filtered=filtered,
                    retrieved=retrieved,
                    recall=recall_at_k(retrieved, question.relevant),
                    reciprocal_rank=reciprocal_rank(retrieved, question.relevant),
                    top_score=search.chunks[0].score if search.chunks else 0.0,
                    confidence=search.retrieval_confidence,
                )
            )
    off_topic = [await retriever.search(text, limit=eval_set.k) for text in eval_set.off_topic]

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 3)

    filtered = [r for r in results if r.filtered]
    unfiltered = [r for r in results if not r.filtered]
    return EvalReport(
        k=eval_set.k,
        filtered_recall=mean([r.recall for r in filtered]),
        unfiltered_recall=mean([r.recall for r in unfiltered]),
        filtered_mrr=mean([r.reciprocal_rank for r in filtered]),
        unfiltered_mrr=mean([r.reciprocal_rank for r in unfiltered]),
        results=tuple(results),
        answerable_confidences=tuple(r.confidence for r in filtered),
        answerable_top_scores=tuple(r.top_score for r in filtered),
        off_topic_confidences=tuple(search.retrieval_confidence for search in off_topic),
        off_topic_top_scores=tuple(search.chunks[0].score if search.chunks else 0.0 for search in off_topic),
    )
