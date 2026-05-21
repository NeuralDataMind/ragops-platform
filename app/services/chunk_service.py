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