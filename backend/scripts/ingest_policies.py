"""Ingest the policy wording PDFs into Qdrant. Idempotent: unchanged documents are skipped.

Usage:
    python -m scripts.ingest_policies
"""

import asyncio
import sys

from app.core.config import get_settings
from app.db.vector import create_qdrant_client
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import PolicyRetriever


async def run() -> int:
    """Ingest every product's wording into the configured collection.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    client = create_qdrant_client(settings)
    try:
        retriever = PolicyRetriever(client, get_embedder(settings.embedding_model_name), settings.qdrant_collection)
        results = await retriever.ingest_policy_documents(settings.policy_documents_dir)
    finally:
        await client.close()
    for result in results:
        print(f"ingest: {result.document}: {result.status} ({result.page_count} pages, {result.chunk_count} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
