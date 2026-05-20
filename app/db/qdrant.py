from qdrant_client import AsyncQdrantClient
from app.core.config import settings

qdrant_client = AsyncQdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)

async def check_qdrant() -> bool:
    try:
        await qdrant_client.get_collections()
        return True
    except Exception:
        return False