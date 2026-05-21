from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.core.config import settings

qdrant_client = AsyncQdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)

def get_collection_name(index_version: str = "index_v1") -> str:
    return f"{settings.QDRANT_COLLECTION_PREFIX}_{index_version}"

async def check_qdrant() -> bool:
    try:
        await qdrant_client.get_collections()
        return True
    except Exception:
        return False
    
async def ensure_collection(index_version: str = "index_v1") -> str:
    collection_name = get_collection_name(index_version)

    collection = await qdrant_client.get_collections()
    existing_names = {collection.name for collection in collection.collections}

    if collection_name not in existing_names:
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

    return collection_name

async def upsert_embedded_chunks(
        embedded_chunks: list[dict[str, Any]],
        index_version: str = "index_v1",
) -> int:
    if not embedded_chunks:
        return 0
    
    collection_name = await ensure_collection(index_version)

    points: list[PointStruct] = []

    for chunk in embedded_chunks:
        points.append(
            PointStruct(
                id=str(chunk["id"]),
                vector=chunk["embedding"],
                payload={
                    "chunk_id": str(chunk["id"]),
                    "document_id": str(chunk["document_id"]),
                    "page_number": chunk.get("page_number"),
                    "chunk_index": chunk["chunk_index"],
                    "index_version": index_version,
                    "embedding_model": chunk["embedding_model"],
                },
            )
        )

    await qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    return len(points)