import json
import time
from typing import Any
from uuid import UUID

from app.db.postgres import get_postgres_pool


def now_ms() -> int:
    return int(time.time() * 1000)


async def create_query_log(
    query: str,
    answer: str,
    refused: bool,
    reason: str | None,
    retrieved_count: int,
    latency_ms: int,
    document_id: UUID | None = None,
) -> dict:
    pool = await get_postgres_pool()

    query_sql = """
    INSERT INTO query_logs (
        query,
        answer,
        refused,
        refusal_reason,
        retrieved_count,
        latency_ms,
        document_id
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    RETURNING
        id,
        query,
        answer,
        refused,
        refusal_reason,
        retrieved_count,
        latency_ms,
        document_id,
        created_at;
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query_sql,
            query,
            answer,
            refused,
            reason,
            retrieved_count,
            latency_ms,
            document_id,
        )

    return dict(row)


async def create_retrieved_chunk_logs(
    query_log_id: UUID,
    retrieved_chunks: list[dict[str, Any]],
) -> int:
    if not retrieved_chunks:
        return 0

    pool = await get_postgres_pool()

    query_sql = """
    INSERT INTO retrieved_chunks_log (
        query_log_id,
        chunk_id,
        document_id,
        chunk_index,
        page_number,
        dense_score,
        sparse_score,
        hybrid_score,
        rank
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);
    """

    async with pool.acquire() as conn:
        async with conn.transaction():
            for rank, chunk in enumerate(retrieved_chunks, start=1):
                await conn.execute(
                    query_sql,
                    query_log_id,
                    UUID(str(chunk["chunk_id"])),
                    UUID(str(chunk["document_id"])),
                    chunk.get("chunk_index"),
                    chunk.get("page_number"),
                    chunk.get("dense_score"),
                    chunk.get("sparse_score"),
                    chunk.get("hybrid_score"),
                    rank,
                )

    return len(retrieved_chunks)