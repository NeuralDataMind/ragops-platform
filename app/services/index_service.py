from uuid import UUID

from app.core.config import settings
from app.db.qdrant import upsert_embedded_chunks
from app.services.chunk_service import (
    get_chunks_by_document_id,
    update_chunks_index_metadata,
)
from app.services.document_service import update_document_status
from app.services.embeddings import embed_chunks_in_batches


async def index_document_chunks(
    document_id: UUID,
    index_version: str = "v1",
) -> dict:
    await update_document_status(document_id, "indexing")

    chunks = await get_chunks_by_document_id(document_id)

    if not chunks:
        await update_document_status(document_id, "indexing_failed")
        raise ValueError("No chunks found for document. Chunk the document before indexing.")

    try:
        embedded_chunks = await embed_chunks_in_batches(
            chunks=chunks,
            batch_size=16,
            delay_seconds=0.5,
        )

        vectors_upserted = await upsert_embedded_chunks(
            embedded_chunks=embedded_chunks,
            index_version=index_version,
        )

        chunks_updated = await update_chunks_index_metadata(
            document_id=document_id,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            index_version=index_version,
        )

        await update_document_status(document_id, "indexed")

        return {
            "document_id": document_id,
            "status": "indexed",
            "index_version": index_version,
            "chunks_embedded": len(embedded_chunks),
            "vectors_upserted": vectors_upserted,
            "chunks_updated": chunks_updated,
        }

    except Exception:
        await update_document_status(document_id, "indexing_failed")
        raise