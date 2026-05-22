from uuid import UUID

from app.core.config import settings
from app.db.qdrant import search_similar_chunks
from app.services.chunk_service import get_chunks_by_ids, keyword_search_chunks
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

def normalize_scores(items: list[dict], score_key: str) -> dict[str, float]:
    if not items:
        return {}
    
    scores = [float(item.get(score_key) or 0.0) for item in items]
    max_score = max(scores)

    if max_score <= 0:
        return {str(item["chunk_id"] if "chunk_id" in item else item["id"]): 0.0 for item in items}
    
    normalized = {}

    for item in items:
        item_id = str(item["chunk_id"] if "chunk_id" in item else item["id"])
        normalized[item_id] = float(item.get(score_key) or 0.0) /  max_score

    return normalized

async def hybrid_retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    index_version: str = "v1",
    document_id: UUID | None = None,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    query_vector = await embed_single_text(query)

    dense_results = await search_similar_chunks(
        query_vector=query_vector,
        top_k=top_k,
        index_version=index_version,
        document_id=str(document_id) if document_id else None,
    )

    sparse_query = expand_sparse_query(query)

    sparse_results = await keyword_search_chunks(
        query_text=sparse_query,
        top_k=top_k,
        document_id=document_id,
    )

    dense_norm = normalize_scores(dense_results, "score")
    sparse_norm = normalize_scores(sparse_results, "sparse_score")

    chunk_ids = set(dense_norm.keys()) | set(sparse_norm.keys())

    sparse_by_id = {str(row["id"]): row for row in sparse_results}

    dense_chunk_ids = [UUID(row["chunk_id"]) for row in dense_results]
    missing_from_sparse = [
        UUID(chunk_id)
        for chunk_id in chunk_ids
        if chunk_id not in sparse_by_id
    ]

    postgres_chunks = await get_chunks_by_ids(dense_chunk_ids + missing_from_sparse)

    chunks_by_id = {str(chunk["id"]): chunk for chunk in postgres_chunks}

    merged: list[dict] = []

    for chunk_id in chunk_ids:
        chunk = sparse_by_id.get(chunk_id) or chunks_by_id.get(chunk_id)

        if not chunk:
            continue

        dense_score = dense_norm.get(chunk_id, 0.0)
        sparse_score = sparse_norm.get(chunk_id, 0.0)

        hybrid_score = (dense_weight * dense_score) + (sparse_weight * sparse_score)

        merged.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(chunk["document_id"]),
                "chunk_text": chunk["chunk_text"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "hybrid_score": hybrid_score,
                "index_version": chunk["index_version"],
            }
        )

    merged.sort(key=lambda row: row["hybrid_score"], reverse=True)

    return merged[:top_k]

def expand_sparse_query(query: str) -> str:
    lowered = query.lower()

    expansions = [query]

    if "first" in lowered and ("5" in lowered or "five" in lowered):
        expansions.append("warmup 0-5 0–5 first five minutes beginning interview")

    if "project walkthrough" in lowered or "5 15" in lowered:
        expansions.append("project walkthrough 5-15 5–15 repo github architecture")

    if "decision" in lowered or "deep dive" in lowered:
        expansions.append("decision deep dive tradeoffs alternatives architecture")

    return " ".join(expansions)