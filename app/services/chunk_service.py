from uuid import UUID

from app.db.postgres import get_postgres_pool
from app.services.chunking import ChunkCandidate


async def create_chunks(chunks: list[ChunkCandidate]) -> list[dict]:
    pool = await get_postgres_pool()

    query = """
    INSERT INTO chunks (
        id,
        document_id,
        chunk_text,
        page_number,
        chunk_index,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
        embedding_model,
        embedding_dimension,
        index_version
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    RETURNING
        id,
        document_id,
        chunk_text,
        page_number,
        chunk_index,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
        embedding_model,
        embedding_dimension,
        index_version,
        created_at;
    """

    created_chunks: list[dict] = []

    async with pool.acquire() as conn:
        async with conn.transaction():
            for chunk in chunks:
                row = await conn.fetchrow(
                    query,
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_text,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.chunking_strategy,
                    chunk.chunk_size,
                    chunk.chunk_overlap,
                    None,
                    None,
                    None,
                )

                created_chunks.append(dict(row))

    return created_chunks


async def get_chunks_by_document_id(document_id: UUID) -> list[dict]:
    pool = await get_postgres_pool()

    query = """
    SELECT
        id,
        document_id,
        chunk_text,
        page_number,
        chunk_index,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
        embedding_model,
        embedding_dimension,
        index_version,
        created_at
    FROM chunks
    WHERE document_id = $1
    ORDER BY chunk_index;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, document_id)

    return [dict(row) for row in rows]


async def update_chunks_index_metadata(
    document_id: UUID,
    embedding_model: str,
    embedding_dimension: int,
    index_version: str,
) -> int:
    pool = await get_postgres_pool()

    query = """
    UPDATE chunks
    SET
        embedding_model = $2,
        embedding_dimension = $3,
        index_version = $4
    WHERE document_id = $1;
    """

    async with pool.acquire() as conn:
        result = await conn.execute(
            query,
            document_id,
            embedding_model,
            embedding_dimension,
            index_version,
        )

    return int(result.split()[-1])