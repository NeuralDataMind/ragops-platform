from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    filename: str = Field(..., min_length=1)
    file_type: str = Field(..., min_length=1)
    source: str = "upload"


class DocumentCreate(DocumentBase):
    storage_path: str | None = None
    status: str = "uploaded"


class DocumentResponse(DocumentBase):
    id: UUID
    status: str
    storage_path: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    count: int

class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    storage_path: str
    status: str
    message: str