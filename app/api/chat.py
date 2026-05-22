from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval import hybrid_retrieve_relevant_chunks
from app.services.generation import generate_grounded_answer


router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5
    document_id: UUID | None = None


@router.post("/answer")
async def answer_question(payload: ChatRequest):
    try:
        retrieved_chunks = await hybrid_retrieve_relevant_chunks(
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
        )

        result = await generate_grounded_answer(
            query=payload.query,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "query": payload.query,
            "answer": result["answer"],
            "citations": result["citations"],
            "refused": result["refused"],
            "retrieved_count": len(retrieved_chunks),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {repr(exc)}",
        )