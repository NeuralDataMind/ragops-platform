from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.services.retrieval import hybrid_retrieve_relevant_chunks

router = APIRouter()


@router.get("/retrieve")
async def retrieve(
    query: str,
    top_k: int = 5,
    document_id: UUID | None = None,
):
    try:
        chunks = await hybrid_retrieve_relevant_chunks(
            query=query,
            top_k=top_k,
            document_id=document_id,
        )

        return {
            "query": query,
            "results": chunks,
            "count": len(chunks),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {repr(exc)}",
        )