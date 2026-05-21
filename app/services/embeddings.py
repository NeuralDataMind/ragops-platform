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
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.EMBEDDING_DIMENSION,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def embed_single_text(text: str) -> list[float]:
    clean_text = text.strip()

    if not clean_text:
        raise EmbeddingError("Cannot embed empty text")

    client = get_embedding_client()

    try:
        embedding = await client.aembed_query(clean_text)
    except Exception as exc:
        raise EmbeddingError(
            f"Embedding single text failed: {repr(exc)}"
        ) from exc

    if len(embedding) != settings.EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"Embedding dimension mismatch. "
            f"Expected {settings.EMBEDDING_DIMENSION}, got {len(embedding)}"
        )

    return embedding


async def embed_text_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds texts one by one.

    We intentionally avoid aembed_documents() for now because LangChain's
    batch path is failing with gemini-embedding-2-preview in this environment.
    """

    if not texts:
        return []

    embeddings: list[list[float]] = []

    for text in texts:
        vector = await embed_single_text(text)
        embeddings.append(vector)

        # Small throttle to reduce rate-limit pressure.
        await asyncio.sleep(0.2)

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