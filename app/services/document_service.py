from uuid import UUID

from app.db.postgres import get_postgres_pool
from app.schemas.document import DocumentCreate


async def create_document(document: DocumentCreate) -> dict:
    pool = await get_postgres_pool()

    query = """
    INSERT INTO documents (
        filename,
        file_type,
        source,
        status,
        storage_path
    )
    VALUES ($1, $2, $3, $4, $5)
    RETURNING
        id,
        filename,
        file_type,
        source,
        status,
        storage_path,
        created_at,
        updated_at;
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            query,
            document.filename,
            document.file_type,
            document.source,
            document.status,
            document.storage_path,
        )

    return dict(row)


async def list_documents() -> list[dict]:
    pool = await get_postgres_pool()

    query = """
    SELECT
        id,
        filename,
        file_type,
        source,
        status,
        storage_path,
        created_at,
        updated_at
    FROM documents
    ORDER BY created_at DESC;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    return [dict(row) for row in rows]


async def get_document_by_id(document_id: UUID) -> dict | None:
    pool = await get_postgres_pool()

    query = """
    SELECT
        id,
        filename,
        file_type,
        source,
        status,
        storage_path,
        created_at,
        updated_at
    FROM documents
    WHERE id = $1;
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, document_id)

    return dict(row) if row else None


async def update_document_status(document_id: UUID, status: str) -> dict | None:
    pool = await get_postgres_pool()

    query = """
    UPDATE documents
    SET
        status = $2,
        updated_at = NOW()
    WHERE id = $1
    RETURNING
        id,
        filename,
        file_type,
        source,
        status,
        storage_path,
        created_at,
        updated_at;
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, document_id, status)

    return dict(row) if row else None