"""GET /api/policies/search: semantic search across the indexed policy wordings (the Policy explorer screen)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_policy_retriever
from app.api.problems import ProblemError
from app.api.security import require_api_key
from app.domain.enums import ProductType
from app.retrieval.retriever import PolicyRetriever
from app.schemas.policies import PolicyPassage, PolicySearchResponse

router = APIRouter(prefix="/policies", tags=["policies"], dependencies=[Depends(require_api_key)])


@router.get("/search", response_model=PolicySearchResponse, summary="Search policy wordings with citations")
async def search_policies(
    retriever: Annotated[PolicyRetriever | None, Depends(get_policy_retriever)],
    q: Annotated[str, Query(min_length=3, max_length=300, description="Question in plain English")],
    product_type: ProductType | None = None,
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
) -> PolicySearchResponse:
    """Return the best-matching passages, each with document and page.

    Args:
        retriever: Policy retriever (None when the embedding model is not loaded).
        q: The question.
        product_type: Restrict to one product's wording.
        limit: Maximum passages.

    Returns:
        Passages with citations and a calibrated retrieval confidence.
    """
    if retriever is None:
        raise ProblemError(503, "search-unavailable", "Search unavailable", "The embedding model is not loaded.")
    result = await retriever.search(q, product_type=product_type, limit=limit)
    return PolicySearchResponse(
        query=q,
        product_type=product_type,
        retrieval_confidence=result.retrieval_confidence,
        passages=[
            PolicyPassage(
                chunk_id=chunk.chunk_id,
                document=chunk.citation.document,
                page=chunk.citation.page,
                section=chunk.citation.section,
                section_kind=chunk.section_kind,
                label=chunk.citation.label,
                text=chunk.text,
                similarity=round(chunk.score, 4),
            )  # fmt: skip
            for chunk in result.chunks
        ],
    )
