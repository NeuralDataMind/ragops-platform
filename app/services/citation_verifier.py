import re
from typing import Any


CHUNK_ID_PATTERNS = [
    # Official format:
    # [chunk_id: ddb6d255-1b89-4a1d-9e26-132afca5bb82]
    re.compile(r"\[chunk_id:\s*([0-9a-fA-F-]{36})\]"),

    # Fallback format sometimes produced by LLM:
    # [ddb6d255-1b89-4a1d-9e26-132afca5bb82]
    re.compile(r"\[([0-9a-fA-F-]{36})\]"),
]


class CitationVerificationError(Exception):
    """Raised when citation verification fails."""


def extract_cited_chunk_ids(answer: str) -> list[str]:
    """
    Extract cited chunk IDs from an answer.

    Supports:
    - [chunk_id: <uuid>]
    - [<uuid>]
    """

    matches: list[str] = []

    for pattern in CHUNK_ID_PATTERNS:
        matches.extend(pattern.findall(answer))

    seen = set()
    cited_ids: list[str] = []

    for chunk_id in matches:
        normalized = chunk_id.lower()

        if normalized not in seen:
            cited_ids.append(normalized)
            seen.add(normalized)

    return cited_ids


def normalize_citation_format(answer: str) -> str:
    """
    Normalize plain UUID citations into the official format.

    Converts:
    [ddb6d255-1b89-4a1d-9e26-132afca5bb82]

    Into:
    [chunk_id: ddb6d255-1b89-4a1d-9e26-132afca5bb82]
    """

    return re.sub(
        r"\[([0-9a-fA-F-]{36})\]",
        r"[chunk_id: \1]",
        answer,
    )


def verify_answer_citations(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict:
    """
    Verifies that every cited chunk_id exists in retrieved_chunks.

    This prevents fake citations like:
    [chunk_id: made-up-id]

    It only accepts citations that came from the retrieval set.
    """

    cited_chunk_ids = extract_cited_chunk_ids(answer)

    retrieved_by_id = {
        str(chunk["chunk_id"]).lower(): chunk
        for chunk in retrieved_chunks
    }

    invalid_citations = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id not in retrieved_by_id
    ]

    if invalid_citations:
        raise CitationVerificationError(
            f"Answer cited chunks that were not retrieved: {invalid_citations}"
        )

    used_citations = [
        {
            "document_id": retrieved_by_id[chunk_id]["document_id"],
            "chunk_id": retrieved_by_id[chunk_id]["chunk_id"],
            "page_number": retrieved_by_id[chunk_id].get("page_number"),
            "chunk_index": retrieved_by_id[chunk_id].get("chunk_index"),
            "dense_score": retrieved_by_id[chunk_id].get("dense_score"),
            "sparse_score": retrieved_by_id[chunk_id].get("sparse_score"),
            "hybrid_score": retrieved_by_id[chunk_id].get("hybrid_score"),
        }
        for chunk_id in cited_chunk_ids
    ]

    return {
        "cited_chunk_ids": cited_chunk_ids,
        "used_citations": used_citations,
        "has_citations": bool(cited_chunk_ids),
        "invalid_citations": invalid_citations,
    }