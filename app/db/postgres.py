import asyncpg
from app.core.config import settings

_pool: asyncpg.Pool | None = None

async def connect_postgres() -> None:
    global _pool

    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=1,
            max_size=10,
        )

async def close_postgres() -> None:
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None

async def get_postgres_pool() -> asyncpg.Pool:
    if _pool is None:
        await connect_postgres()
    
    assert _pool is not None
    return _pool

async def check_postgres() -> bool:
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
        return result == 1
    except Exception as exc:
        return False