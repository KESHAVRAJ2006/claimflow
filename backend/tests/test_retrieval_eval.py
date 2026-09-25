"""Retrieval quality gate with the real embedding model: Recall@5 >= 0.8 and calibrated confidence."""

import pytest

from app.core.config import get_settings
from app.retrieval.embeddings import get_embedder
from app.rules.thresholds import MIN_DECISION_CONFIDENCE
from scripts.eval_retrieval import TARGET_RECALL, evaluate

pytestmark = pytest.mark.model


@pytest.fixture(scope="module", autouse=True)
def _model_available() -> None:
    try:
        get_embedder(get_settings().embedding_model_name)
    except Exception as exc:  # noqa: BLE001 — offline without a cached model, etc.
        pytest.skip(f"embedding model unavailable: {exc}")


async def test_recall_at_5_meets_target_with_and_without_product_filter() -> None:
    report = await evaluate()
    assert len(report.results) >= 20  # >= 10 questions, each in two modes
    assert report.filtered_recall >= TARGET_RECALL
    assert report.unfiltered_recall >= TARGET_RECALL


async def test_confidence_separates_answerable_from_off_topic_questions() -> None:
    report = await evaluate()
    threshold = float(MIN_DECISION_CONFIDENCE)
    # Answerable questions should normally clear the escalation threshold; off-topic ones must never do so.
    assert report.median_answerable_confidence >= threshold
    assert report.max_off_topic_confidence < threshold
