"""Evaluate retrieval quality: Recall@5 and MRR on labelled questions, plus confidence calibration.

Ingests the policy PDFs into an in-memory Qdrant with the real embedding model, so results are reproducible
and never affected by whatever the shared Qdrant server holds.

Usage:
    python -m scripts.eval_retrieval
Exit code 1 if Recall@5 (with or without the product filter) is below the target.
"""

import asyncio
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.retrieval.embeddings import get_embedder
from app.retrieval.evaluation import EvalReport, load_eval_set, run_retrieval_eval
from app.retrieval.retriever import PolicyRetriever
from app.rules.thresholds import MIN_DECISION_CONFIDENCE

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "retrieval_questions.json"
TARGET_RECALL = 0.8


async def evaluate(eval_path: Path = EVAL_SET_PATH) -> EvalReport:
    """Build an in-memory index of the policy PDFs and run the evaluation.

    Args:
        eval_path: Labelled question file.

    Returns:
        The evaluation report.
    """
    settings = get_settings()
    client = AsyncQdrantClient(location=":memory:")
    try:
        retriever = PolicyRetriever(client, get_embedder(settings.embedding_model_name), "policy_chunks_eval")
        await retriever.ingest_policy_documents(settings.policy_documents_dir)
        return await run_retrieval_eval(retriever, load_eval_set(eval_path))
    finally:
        await client.close()


def print_report(report: EvalReport) -> None:
    """Print the report as Markdown.

    Args:
        report: Evaluation report.
    """
    print(f"| Question | Filtered R@{report.k} | RR | Top score | Confidence | Unfiltered R@{report.k} | RR |")
    print("|---|---|---|---|---|---|---|")
    by_id: dict[str, dict[bool, object]] = {}
    for result in report.results:
        by_id.setdefault(result.id, {})[result.filtered] = result
    for question_id, modes in by_id.items():
        f, u = modes[True], modes[False]
        print(
            f"| {question_id} | {f.recall:.2f} | {f.reciprocal_rank:.2f} | {f.top_score:.3f} | {f.confidence:.2f} "  # type: ignore[attr-defined]
            f"| {u.recall:.2f} | {u.reciprocal_rank:.2f} |"  # type: ignore[attr-defined]
        )
    print()
    k = report.k
    print(f"Recall@{k} with product filter:    {report.filtered_recall:.3f}  (MRR {report.filtered_mrr:.3f})")
    print(f"Recall@{k} without product filter: {report.unfiltered_recall:.3f}  (MRR {report.unfiltered_mrr:.3f})")
    print(
        f"Retrieval confidence: answerable median {report.median_answerable_confidence:.2f}, "
        f"min {min(report.answerable_confidences):.2f}; off-topic max {report.max_off_topic_confidence:.2f} "
        f"(escalation threshold {MIN_DECISION_CONFIDENCE})"
    )
    print(
        f"Raw top similarity: answerable {min(report.answerable_top_scores):.3f}-"
        f"{max(report.answerable_top_scores):.3f}; off-topic max {max(report.off_topic_top_scores, default=0.0):.3f}"
    )


def main() -> int:
    """CLI entry point.

    Returns:
        0 if both recall figures meet the target, otherwise 1.
    """
    report = asyncio.run(evaluate())
    print_report(report)
    passed = report.filtered_recall >= TARGET_RECALL and report.unfiltered_recall >= TARGET_RECALL
    print(f"\n{'PASS' if passed else 'FAIL'}: target Recall@{report.k} >= {TARGET_RECALL}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
