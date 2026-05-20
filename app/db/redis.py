from redis.asyncio import Redis
from app.core.config import settings

redis_client = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
)

async def check_redis() -> bool:
    try:
        reponse = await redis_client.ping()
        return reponse is True
    except Exception:
        return False