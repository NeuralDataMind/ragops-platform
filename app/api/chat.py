import time

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval import hybrid_retrieve_relevant_chunks
from app.services.generation import generate_grounded_answer
from app.services.monitoring import (
    create_query_log,
    create_retrieved_chunk_logs,
)

from app.core.config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5
    document_id: UUID | None = None


@router.post("/answer")
async def answer_question(payload: ChatRequest):
    start_time = time.perf_counter()

    try:
        retrieved_chunks = await hybrid_retrieve_relevant_chunks(
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
        )

        if not retrieved_chunks:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            response_payload = {
                "query": payload.query,
                "answer": "I do not have enough retrieved evidence to answer this question.",
                "citations": [],
                "refused": True,
                "reason": "no_retrieved_chunks",
                "retrieved_count": 0,
            }

            query_log = await create_query_log(
                query=payload.query,
                answer=response_payload["answer"],
                refused=True,
                reason="no_retrieved_chunks",
                retrieved_count=0,
                latency_ms=latency_ms,
                document_id=payload.document_id,
            )

            return {
                **response_payload,
                "latency_ms": latency_ms,
                "query_log_id": str(query_log["id"]),
            }

        top_score = retrieved_chunks[0].get("hybrid_score", 0)

        if top_score < settings.MIN_HYBRID_SCORE:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            response_payload = {
                "query": payload.query,
                "answer": "I do not have enough high-confidence evidence to answer this question.",
                "citations": [],
                "refused": True,
                "reason": "low_retrieval_confidence",
                "top_hybrid_score": top_score,
                "threshold": settings.MIN_HYBRID_SCORE,
                "retrieved_count": len(retrieved_chunks),
            }

            query_log = await create_query_log(
                query=payload.query,
                answer=response_payload["answer"],
                refused=True,
                reason="low_retrieval_confidence",
                retrieved_count=len(retrieved_chunks),
                latency_ms=latency_ms,
                document_id=payload.document_id,
            )

            await create_retrieved_chunk_logs(
                query_log_id=query_log["id"],
                retrieved_chunks=retrieved_chunks,
            )

            return {
                **response_payload,
                "latency_ms": latency_ms,
                "query_log_id": str(query_log["id"]),
            }

        result = await generate_grounded_answer(
            query=payload.query,
            retrieved_chunks=retrieved_chunks,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        response_payload = {
            "query": payload.query,
            "answer": result["answer"],
            "citations": result["citations"],
            "refused": result["refused"],
            "retrieved_count": len(retrieved_chunks),
        }

        if "reason" in result:
            response_payload["reason"] = result["reason"]

        query_log = await create_query_log(
            query=payload.query,
            answer=response_payload["answer"],
            refused=response_payload["refused"],
            reason=response_payload.get("reason"),
            retrieved_count=len(retrieved_chunks),
            latency_ms=latency_ms,
            document_id=payload.document_id,
        )

        await create_retrieved_chunk_logs(
            query_log_id=query_log["id"],
            retrieved_chunks=retrieved_chunks,
        )

        return {
            **response_payload,
            "latency_ms": latency_ms,
            "query_log_id": str(query_log["id"]),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {repr(exc)}",
        )