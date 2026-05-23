from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.citation_verifier import (
    normalize_citation_format,
    verify_answer_citations,
)


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


def is_insufficient_evidence_answer(answer_text: str) -> bool:
    normalized = answer_text.strip().lower()

    refusal_phrases = [
        "insufficient_evidence",
        "do not have enough evidence",
        "do not have enough cited evidence",
        "does not contain",
        "not contain any information",
        "no information about",
        "cannot answer",
    ]

    return any(phrase in normalized for phrase in refusal_phrases)


def attach_top_chunk_citation_if_safe(
    answer_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str | None:
    """
    Deterministic fallback.

    If the model gives a useful answer but forgets citation syntax,
    auto-attach the top chunk citation only when retrieval confidence is strong.
    """

    if not retrieved_chunks:
        return None

    top_chunk = retrieved_chunks[0]

    hybrid_score = float(top_chunk.get("hybrid_score") or 0)
    sparse_score = float(top_chunk.get("sparse_score") or 0)

    if hybrid_score < settings.MIN_HYBRID_SCORE:
        return None

    # Require sparse support so we do not auto-cite purely semantic matches.
    if sparse_score <= 0:
        return None

    cleaned_answer = answer_text.strip()

    if not cleaned_answer:
        return None

    chunk_id = top_chunk["chunk_id"]

    return f"{cleaned_answer} [chunk_id: {chunk_id}]"


async def generate_with_prompt(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return str(response.content).strip()


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
3. Every answer sentence must end with at least one citation in this exact format: [chunk_id: <chunk_id>].
4. Use only chunk IDs that appear in the retrieved context.
5. If the context does not contain the answer, output exactly: INSUFFICIENT_EVIDENCE.
6. Do not explain why evidence is missing unless you are directly answering with evidence.
7. Do not invent details.
""".strip()

    user_prompt = f"""
User question:
{query}

Retrieved context:
{context_block}

Write a concise answer grounded only in the retrieved context.
Every answer sentence must include a valid chunk_id citation.

If the retrieved context does not answer the question, output exactly:
INSUFFICIENT_EVIDENCE
""".strip()

    answer_text = await generate_with_prompt(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    if is_insufficient_evidence_answer(answer_text):
        return {
            "answer": "I do not have enough cited evidence to answer this question.",
            "citations": [],
            "refused": True,
            "reason": "insufficient_evidence",
        }

    verification = verify_answer_citations(
        answer=answer_text,
        retrieved_chunks=retrieved_chunks,
    )

    if verification["has_citations"]:
        return {
            "answer": normalize_citation_format(answer_text),
            "citations": verification["used_citations"],
            "refused": False,
        }

    retry_prompt = f"""
Your previous answer did not include valid citations.

You must answer using ONLY the retrieved context.

Mandatory citation format:
[chunk_id: <exact_chunk_id>]

Use at least one citation from the retrieved context.
Every answer sentence must end with a valid chunk_id citation.

If the context does not answer the question, output exactly:
INSUFFICIENT_EVIDENCE

User question:
{query}

Retrieved context:
{context_block}
""".strip()

    retry_answer_text = await generate_with_prompt(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=retry_prompt,
    )

    if is_insufficient_evidence_answer(retry_answer_text):
        return {
            "answer": "I do not have enough cited evidence to answer this question.",
            "citations": [],
            "refused": True,
            "reason": "insufficient_evidence",
        }

    retry_verification = verify_answer_citations(
        answer=retry_answer_text,
        retrieved_chunks=retrieved_chunks,
    )

    if retry_verification["has_citations"]:
        return {
            "answer": normalize_citation_format(retry_answer_text),
            "citations": retry_verification["used_citations"],
            "refused": False,
        }

    fallback_answer = attach_top_chunk_citation_if_safe(
        answer_text=retry_answer_text,
        retrieved_chunks=retrieved_chunks,
    )

    if fallback_answer:
        fallback_verification = verify_answer_citations(
            answer=fallback_answer,
            retrieved_chunks=retrieved_chunks,
        )

        if fallback_verification["has_citations"]:
            return {
                "answer": normalize_citation_format(fallback_answer),
                "citations": fallback_verification["used_citations"],
                "refused": False,
                "citation_source": "auto_attached_from_top_retrieval",
            }

    return {
        "answer": "I do not have enough cited evidence to answer this question.",
        "citations": [],
        "refused": True,
        "reason": "answer_missing_citations",
    }