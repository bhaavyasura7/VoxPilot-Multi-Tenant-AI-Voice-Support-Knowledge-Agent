from pydantic import BaseModel
from datetime import datetime


class DocumentResponse(BaseModel):
    id: int
    tenant_id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    error_message: str | None = None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    status: str
    message: str
