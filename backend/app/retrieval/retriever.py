"""Qdrant-backed policy retrieval: idempotent ingestion and filtered semantic search."""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field
from qdrant_client import AsyncQdrantClient, models

from app.domain.enums import ProductType
from app.domain.products import POLICY_DOCUMENTS
from app.retrieval.chunking import Chunk, SectionKind, chunk_document
from app.retrieval.embeddings import Embedder
from app.retrieval.pdf_tools import extract_pdf_file

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
UPSERT_BATCH_SIZE = 64
# Payload fields the search filters on; indexing them keeps filtered search fast as the collection grows.
_INDEXED_FIELDS = ("document", "product_type", "section_kind")

# Cosine similarity is not a probability. These two anchors map it onto a 0-1 retrieval confidence. Measured
# with `python -m scripts.eval_retrieval` (all-MiniLM-L6-v2, heading-aware chunks): the best hit for
# answerable questions scores 0.43-0.72 and for off-topic questions at most 0.11, so the floor sits clear of
# off-topic noise. All 12 answerable questions clear the 0.65 escalation line, the lowest only just (0.66):
# a question phrased further from the wording will escalate, which is the intended way for retrieval to fail.
CONFIDENCE_SCORE_FLOOR = 0.20  # at or below this similarity, confidence is 0
CONFIDENCE_SCORE_CEILING = 0.55  # at or above this similarity, confidence is 1


class Citation(BaseModel):
    """Where a passage came from."""

    model_config = ConfigDict(frozen=True)

    document: str
    page: int
    section: str | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        """Short human-readable form, e.g. ``Motor_Policy.pdf p.3``."""
        return f"{self.document} p.{self.page}"


class RetrievedChunk(BaseModel):
    """One search hit."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    citation: Citation
    section_kind: SectionKind
    product_type: ProductType
    score: float = Field(description="Cosine similarity between query and chunk")
    char_start: int
    char_end: int


class SearchResult(BaseModel):
    """Ranked hits for one query plus a calibrated confidence."""

    model_config = ConfigDict(frozen=True)

    query: str
    product_type: ProductType | None
    section_kinds: tuple[SectionKind, ...]
    chunks: tuple[RetrievedChunk, ...]
    retrieval_confidence: float = Field(ge=0, le=1)

    def summary(self) -> str:
        """One line for the tool call log shown in the UI.

        Returns:
            e.g. ``5 passages; best Motor_Policy.pdf p.3 (0.61); confidence 1.00``.
        """
        if not self.chunks:
            return "no matching passages; confidence 0.00"
        best = self.chunks[0]
        return (
            f"{len(self.chunks)} passages; best {best.citation.label} ({best.score:.2f}); "
            f"confidence {self.retrieval_confidence:.2f}"
        )


class IngestResult(BaseModel):
    """Outcome of ingesting one document."""

    model_config = ConfigDict(frozen=True)

    document: str
    status: Literal["ingested", "unchanged"]
    page_count: int
    chunk_count: int


def retrieval_confidence(top_score: float | None) -> float:
    """Convert the best cosine similarity into a 0-1 confidence.

    Args:
        top_score: Similarity of the best hit, or None if there were no hits.

    Returns:
        Linear interpolation between the floor and ceiling anchors, clamped to [0, 1], rounded to 3 places.
    """
    if top_score is None:
        return 0.0
    scaled = (top_score - CONFIDENCE_SCORE_FLOOR) / (CONFIDENCE_SCORE_CEILING - CONFIDENCE_SCORE_FLOOR)
    return round(min(1.0, max(0.0, scaled)), 3)


def _embedding_text(chunk: Chunk) -> str:
    """Text that is embedded for a chunk: headings first, so a passage is found by the topic of its section."""
    headings = " > ".join(heading for heading in (chunk.section_title, chunk.subsection_title) if heading)
    return f"{headings}\n{chunk.text}" if headings else chunk.text


class PolicyRetriever:
    """Ingests policy PDFs into Qdrant and searches them."""

    def __init__(self, client: AsyncQdrantClient, embedder: Embedder, collection_name: str) -> None:
        """Bind the retriever to a Qdrant collection.

        Args:
            client: Async Qdrant client.
            embedder: Embedding model (loaded once by the caller).
            collection_name: Collection holding policy chunks.
        """
        self._client = client
        self._embedder = embedder
        self._collection = collection_name

    async def ensure_collection(self) -> None:
        """Create the collection and payload indexes if they don't exist yet.

        Raises:
            ValueError: If the collection exists with a different vector size (e.g. a different model).
        """
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != self._embedder.dimension:
                raise ValueError(
                    f"Collection {self._collection} has vector size {size}, but the embedder produces "
                    f"{self._embedder.dimension}. Delete the collection or use a new collection name."
                )
        else:
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(size=self._embedder.dimension, distance=models.Distance.COSINE),
            )
        for field in _INDEXED_FIELDS:
            # Idempotent: re-creating an existing index is a no-op on the server.
            await self._client.create_payload_index(self._collection, field, models.PayloadSchemaType.KEYWORD)

    async def ingest_pdf(self, path: Path, product_type: ProductType) -> IngestResult:
        """Ingest one policy PDF. Safe to call repeatedly.

        If the collection already holds exactly this version of the document (same SHA-256 and chunk count),
        nothing is embedded or written. If the document changed, its old chunks are deleted first so no stale
        passage can be retrieved.

        Args:
            path: PDF file path.
            product_type: Product the wording belongs to.

        Returns:
            Whether the document was ingested or already up to date.
        """
        document = extract_pdf_file(path)
        chunks = chunk_document(document, product_type)
        same_document = models.Filter(
            must=[
                models.FieldCondition(key="document", match=models.MatchValue(value=document.document)),
                models.FieldCondition(key="document_sha256", match=models.MatchValue(value=document.sha256)),
            ]
        )
        existing = await self._client.count(self._collection, count_filter=same_document, exact=True)
        if existing.count == len(chunks):
            return IngestResult(
                document=document.document, status="unchanged", page_count=document.page_count, chunk_count=len(chunks)
            )

        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document", match=models.MatchValue(value=document.document))]
                )
            ),
            wait=True,
        )
        for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]
            # Embedding is CPU-bound; running it in a thread keeps the event loop responsive.
            vectors = await asyncio.to_thread(self._embedder.embed, [_embedding_text(chunk) for chunk in batch])
            await self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.model_dump(mode="json"))
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
                wait=True,
            )
        logger.info("ingested policy document", extra={"document": document.document, "chunks": len(chunks)})
        return IngestResult(
            document=document.document, status="ingested", page_count=document.page_count, chunk_count=len(chunks)
        )

    async def ingest_policy_documents(self, directory: Path) -> list[IngestResult]:
        """Ingest the wording for every product.

        Args:
            directory: Folder containing the files named in ``POLICY_DOCUMENTS``.

        Returns:
            One result per product document.

        Raises:
            FileNotFoundError: If a product's wording is missing.
        """
        await self.ensure_collection()
        results = []
        for product_type, file_name in POLICY_DOCUMENTS.items():
            path = directory / file_name
            if not path.is_file():
                raise FileNotFoundError(f"Policy wording for {product_type} not found at {path}")
            results.append(await self.ingest_pdf(path, product_type))
        return results

    async def search(
        self,
        query: str,
        *,
        product_type: ProductType | None = None,
        section_kinds: Sequence[SectionKind] = (),
        limit: int = DEFAULT_LIMIT,
    ) -> SearchResult:
        """Semantic search over policy chunks.

        Args:
            query: Natural-language question.
            product_type: Restrict to one product's wording.
            section_kinds: Restrict to these kinds of section (empty means all).
            limit: Maximum hits.

        Returns:
            Hits ordered by similarity, with a calibrated retrieval confidence.
        """
        conditions: list[models.Condition] = []
        if product_type is not None:
            conditions.append(models.FieldCondition(key="product_type", match=models.MatchValue(value=product_type)))
        if section_kinds:
            kinds = [kind.value for kind in section_kinds]
            conditions.append(models.FieldCondition(key="section_kind", match=models.MatchAny(any=kinds)))
        [vector] = await asyncio.to_thread(self._embedder.embed, [query])
        response = await self._client.query_points(
            self._collection,
            query=vector,
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=limit,
            with_payload=True,
        )
        hits = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    text=payload["text"],
                    citation=Citation(
                        document=payload["document"],
                        page=payload["page_number"],
                        section=payload.get("subsection_title") or payload.get("section_title"),
                    ),
                    section_kind=SectionKind(payload["section_kind"]),
                    product_type=ProductType(payload["product_type"]),
                    score=round(point.score, 4),
                    char_start=payload["char_start"],
                    char_end=payload["char_end"],
                )
            )
        return SearchResult(
            query=query,
            product_type=product_type,
            section_kinds=tuple(section_kinds),
            chunks=tuple(hits),
            retrieval_confidence=retrieval_confidence(hits[0].score if hits else None),
        )

    async def search_policy(self, query: str, product_type: ProductType | None = None) -> SearchResult:
        """General search across all sections of the wordings.

        Args:
            query: Natural-language question.
            product_type: Optional product filter.

        Returns:
            The search result.
        """
        return await self.search(query, product_type=product_type)

    async def check_exclusions(self, product_type: ProductType, incident_description: str) -> SearchResult:
        """Find exclusion clauses (and conditions that can void cover) relevant to an incident.

        Args:
            product_type: The claim's product.
            incident_description: What happened, in plain words.

        Returns:
            Hits from exclusion and general-condition sections only.
        """
        return await self.search(
            incident_description,
            product_type=product_type,
            section_kinds=(SectionKind.EXCLUSIONS, SectionKind.CONDITIONS),
        )

    async def get_waiting_period(self, product_type: ProductType, condition_or_event: str) -> SearchResult:
        """Find waiting-period clauses for a condition, procedure or event.

        Args:
            product_type: The claim's product.
            condition_or_event: e.g. "knee arthroscopy" or "cyclone damage".

        Returns:
            Hits from waiting-period sections only.
        """
        return await self.search(
            f"waiting period for {condition_or_event}",
            product_type=product_type,
            section_kinds=(SectionKind.WAITING_PERIODS,),
        )

    async def get_coverage_section(self, product_type: ProductType, incident_type: str) -> SearchResult:
        """Find what the policy covers for a type of incident, including sub-limits and deductibles.

        Args:
            product_type: The claim's product.
            incident_type: e.g. "theft", "cataract surgery", "burst pipe".

        Returns:
            Hits from coverage and limits sections only.
        """
        return await self.search(
            f"cover and limits for {incident_type}",
            product_type=product_type,
            section_kinds=(SectionKind.COVERAGE, SectionKind.LIMITS),
        )
