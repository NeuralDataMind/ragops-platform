import asyncio
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


def get_embedding_client() -> GoogleGenerativeAIEmbeddings:
    if settings.EMBEDDING_PROVIDER != "gemini":
        raise ValueError(
            f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}"
        )

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing")

    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def embed_text_batch(texts: list[str]) -> list[list[float]]:
    
    if not texts:
        return []

    client = get_embedding_client()

    try:
        embeddings = await client.aembed_documents(texts)
    except Exception as exc:
        raise EmbeddingError(f"Embedding batch failed: {repr(exc)}") from exc

    for embedding in embeddings:
        if len(embedding) != settings.EMBEDDING_DIMENSION:
            raise EmbeddingError(
                f"Embedding dimension mismatch. "
                f"Expected {settings.EMBEDDING_DIMENSION}, got {len(embedding)}"
            )

    return embeddings


async def embed_chunks_in_batches(
    chunks: list[dict[str, Any]],
    batch_size: int = 16,
    delay_seconds: float = 0.5,
) -> list[dict[str, Any]]:

    embedded_chunks: list[dict[str, Any]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["chunk_text"] for chunk in batch]

        vectors = await embed_text_batch(texts)

        for chunk, vector in zip(batch, vectors):
            embedded_chunks.append(
                {
                    **chunk,
                    "embedding": vector,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "embedding_dimension": settings.EMBEDDING_DIMENSION,
                }
            )

        if start + batch_size < len(chunks):
            await asyncio.sleep(delay_seconds)

    return embedded_chunks