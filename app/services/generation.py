from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.citation_verifier import verify_answer_citations


class GroundedGenerationError(Exception):
    """Raised when grounded generation fails."""


def get_llm_client() -> ChatGoogleGenerativeAI:
    if settings.LLM_PROVIDER != "gemini":
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing")

    return ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
    )


def build_context_block(retrieved_chunks: list[dict[str, Any]]) -> str:
    context_parts: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
[Citation {index}]
document_id: {chunk["document_id"]}
chunk_id: {chunk["chunk_id"]}
page_number: {chunk.get("page_number")}
chunk_index: {chunk.get("chunk_index")}
text:
{chunk["chunk_text"]}
""".strip()
        )

    return "\n\n".join(context_parts)


async def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict:
    if not query.strip():
        raise GroundedGenerationError("Query cannot be empty")

    if not retrieved_chunks:
        return {
            "answer": "I do not have enough retrieved evidence to answer this question.",
            "citations": [],
            "refused": True,
            "reason": "no_retrieved_chunks",
        }

    llm = get_llm_client()
    context_block = build_context_block(retrieved_chunks)

    system_prompt = """
You are a grounded RAG answer generator.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Every factual claim must be supported by a citation.
4. Cite using [chunk_id: <chunk_id>].
5. If the context does not contain the answer, say you do not have enough evidence.
6. Do not invent details.
""".strip()

    user_prompt = f"""
User question:
{query}

Retrieved context:
{context_block}

Write a concise answer grounded only in the retrieved context.
""".strip()

    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    verification = verify_answer_citations(
        answer=response.content,
        retrieved_chunks=retrieved_chunks,
    )

    if not verification["has_citations"]:
        return {
            "answer": "I do not have enough cited evidence to answer this question.",
            "citations": [],
            "refused": True,
            "reason": "answer_missing_citations",
        }

    return {
        "answer": response.content,
        "citations": verification["used_citations"],
        "refused": False,
    }