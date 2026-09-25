"""Response model for GET /api/policies/search."""

from pydantic import BaseModel

from app.domain.enums import ProductType
from app.retrieval.chunking import SectionKind


class PolicyPassage(BaseModel):
    """One retrieved passage with its citation."""

    chunk_id: str
    document: str
    page: int
    section: str | None
    section_kind: SectionKind
    label: str
    text: str
    similarity: float


class PolicySearchResponse(BaseModel):
    """Search results across the indexed policy wordings."""

    query: str
    product_type: ProductType | None
    retrieval_confidence: float
    passages: list[PolicyPassage]
