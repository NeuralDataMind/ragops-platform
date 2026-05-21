from uuid import UUID

from app.core.config import settings
from app.db.qdrant import search_similar_chunks
from app.services.chunk_service import get_chunks_by_ids
from app.services.embeddings import embed_single_text


async def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
    index_version: str = "v1",
    document_id: UUID | None = None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    top_k = top_k or settings.RERANK_TOP_K

    query_vector = await embed_single_text(query)

    qdrant_results = await search_similar_chunks(
        query_vector=query_vector,
        top_k=top_k,
        index_version=index_version,
        document_id=str(document_id) if document_id else None,
    )

    chunk_ids = [UUID(result["chunk_id"]) for result in qdrant_results]
    postgres_chunks = await get_chunks_by_ids(chunk_ids)

    chunks_by_id = {
        str(chunk["id"]): chunk
        for chunk in postgres_chunks
    }

    retrieved_chunks: list[dict] = []

    for result in qdrant_results:
        chunk = chunks_by_id.get(result["chunk_id"])

        if not chunk:
            continue

        retrieved_chunks.append(
            {
                "chunk_id": result["chunk_id"],
                "document_id": result["document_id"],
                "chunk_text": chunk["chunk_text"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "dense_score": result["score"],
                "index_version": chunk["index_version"],
            }
        )

    return retrieved_chunks