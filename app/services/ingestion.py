from pathlib import Path
from uuid import UUID

from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel


class ExtractedBlock(BaseModel):
    document_id: UUID
    text: str
    source_path: str
    file_type: str
    page_number: int | None = None
    block_index: int
    extraction_method: str


class ExtractedDocument(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    source_path: str
    blocks: list[ExtractedBlock]


def extract_document(document_id: UUID, file_path: str) -> ExtractedDocument:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower().replace(".", "")

    if suffix in ["txt", "md"]:
        return _extract_text_file(document_id, path, suffix)

    if suffix == "pdf":
        return _extract_pdf_file(document_id, path, suffix)

    raise ValueError(f"Unsupported file type for extraction: {suffix}")


def _extract_text_file(
    document_id: UUID,
    path: Path,
    file_type: str,
) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8").strip()

    blocks: list[ExtractedBlock] = []

    if text:
        blocks.append(
            ExtractedBlock(
                document_id=document_id,
                text=text,
                source_path=str(path),
                file_type=file_type,
                page_number=None,
                block_index=0,
                extraction_method="plain_text",
            )
        )

    return ExtractedDocument(
        document_id=document_id,
        filename=path.name,
        file_type=file_type,
        source_path=str(path),
        blocks=blocks,
    )


def _extract_pdf_file(
    document_id: UUID,
    path: Path,
    file_type: str,
) -> ExtractedDocument:
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    blocks: list[ExtractedBlock] = []

    for block_index, page_doc in enumerate(pages):
        text = page_doc.page_content.strip()

        if not text:
            continue

        # PyPDFLoader page index is usually zero-based.
        raw_page_number = page_doc.metadata.get("page")
        page_number = raw_page_number + 1 if isinstance(raw_page_number, int) else None

        blocks.append(
            ExtractedBlock(
                document_id=document_id,
                text=text,
                source_path=str(path),
                file_type=file_type,
                page_number=page_number,
                block_index=block_index,
                extraction_method="langchain_pypdfloader",
            )
        )

    return ExtractedDocument(
        document_id=document_id,
        filename=path.name,
        file_type=file_type,
        source_path=str(path),
        blocks=blocks,
    )