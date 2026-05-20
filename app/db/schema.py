from app.db.postgres import get_postgres_pool


CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
    status TEXT NOT NULL DEFAULT 'uploaded',
    storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    page_number INT,
    chunk_index INT NOT NULL,
    chunking_strategy TEXT NOT NULL,
    chunk_size INT NOT NULL,
    chunk_overlap INT NOT NULL,
    embedding_model TEXT,
    embedding_dimension INT,
    index_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT,
    user_query TEXT NOT NULL,
    answer TEXT,
    used_config_id TEXT,
    latency_ms INT,
    retrieval_latency_ms INT,
    llm_latency_ms INT,
    prompt_tokens INT,
    completion_tokens INT,
    estimated_cost NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieved_chunks_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_log_id UUID NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    dense_score DOUBLE PRECISION,
    sparse_score DOUBLE PRECISION,
    rerank_score DOUBLE PRECISION,
    rank_position INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_index_version ON chunks(index_version);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);
"""


async def init_db_schema() -> None:
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)