from uuid import UUID
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, File, UploadFile

from app.core.config import settings

from app.services.ingestion import extract_document
from app.services.chunking import chunk_extracted_document
from app.services.chunk_service import create_chunks
from app.services.index_service import index_document_chunks 

from app.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)

from app.services.document_service import (
    create_document,
    get_document_by_id,
    list_documents,
    update_document_status,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    original_filename = file.filename

    if not original_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(original_filename).name
    storage_path = upload_dir / safe_filename

    content = await file.read()

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_MB} MB",
        )

    with open(storage_path, "wb") as f:
        f.write(content)

    document_payload = DocumentCreate(
        filename=safe_filename,
        file_type=file_extension.replace(".", ""),
        source="upload",
        status="uploaded",
        storage_path=str(storage_path),
    )

    document = await create_document(document_payload)

    return {
        "document_id": document["id"],
        "filename": document["filename"],
        "file_type": document["file_type"],
        "storage_path": document["storage_path"],
        "status": document["status"],
        "message": "Document uploaded successfully",
    }

@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document_metadata(payload: DocumentCreate):
    document = await create_document(payload)
    return document


@router.get("/", response_model=DocumentListResponse)
async def get_documents():
    documents = await list_documents()
    return {
        "documents": documents,
        "count": len(documents),
    }


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: UUID):
    document = await get_document_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document


@router.patch("/{document_id}/status", response_model=DocumentResponse)
async def patch_document_status(document_id: UUID, new_status: str):
    document = await update_document_status(document_id, new_status)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document

@router.post("/{document_id}/chunk")
async def chunk_document(document_id: UUID):
    document = await get_document_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    storage_path = document.get("storage_path")

    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no storage path",
        )

    try:
        await update_document_status(document_id, "chunking")

        extracted_document = extract_document(
            document_id=document_id,
            file_path=storage_path,
        )

        if not extracted_document.blocks:
            await update_document_status(document_id, "chunking_failed")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No extractable text found in document",
            )

        chunk_candidates = chunk_extracted_document(extracted_document)

        if not chunk_candidates:
            await update_document_status(document_id, "chunking_failed")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks created from document",
            )

        created_chunks = await create_chunks(chunk_candidates)

        await update_document_status(document_id, "chunked")

        return {
            "document_id": document_id,
            "chunks_created": len(created_chunks),
            "status": "chunked",
        }

    except HTTPException:
        raise

    except Exception as exc:
        await update_document_status(document_id, "chunking_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {repr(exc)}",
        )
    
@router.post("/{document_id}/index")
async def index_document(document_id: UUID):
    document = await get_document_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if document["status"] not in ["chunked", "indexing_failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document must be chunked before indexing. Current status: {document['status']}",
        )

    try:
        result = await index_document_chunks(document_id=document_id)
        return result

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {repr(exc)}",
        )