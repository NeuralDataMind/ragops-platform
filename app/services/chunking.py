from uuid import UUID, uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.core.config import settings
from app.services.ingestion import ExtractedDocument


class ChunkCandidate(BaseModel):
    id: UUID
    document_id: UUID
    chunk_text: str
    page_number: int | None = None
    chunk_index: int
    source_path: str
    file_type: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


def chunk_extracted_document(extracted_document: ExtractedDocument) -> list[ChunkCandidate]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunk_candidates: list[ChunkCandidate] = []
    global_chunk_index = 0

    for block in extracted_document.blocks:
        split_texts = splitter.split_text(block.text)

        for split_text in split_texts:
            clean_text = split_text.strip()

            if not clean_text:
                continue

            chunk_candidates.append(
                ChunkCandidate(
                    id=uuid4(),
                    document_id=extracted_document.document_id,
                    chunk_text=clean_text,
                    page_number=block.page_number,
                    chunk_index=global_chunk_index,
                    source_path=block.source_path,
                    file_type=block.file_type,
                    chunking_strategy="recursive_character",
                    chunk_size=settings.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
                )
            )

            global_chunk_index += 1

    return chunk_candidates