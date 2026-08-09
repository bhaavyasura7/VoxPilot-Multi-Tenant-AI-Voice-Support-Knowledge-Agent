from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentStatus
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.auth.deps import get_current_user
from app.services.document_service import validate_file, save_upload
from app.services.pipeline import process_document_by_id

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    file_size = len(content)

    error = validate_file(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        file_size=file_size,
    )
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)

    file_type = "pdf" if "pdf" in (file.content_type or "") else "docx"
    saved = await save_upload(content, file.filename or "unknown", current_user.tenant_id)

    document = Document(
        tenant_id=current_user.tenant_id,
        filename=saved["filename"],
        original_filename=file.filename or "unknown",
        file_type=file_type,
        file_size=saved["file_size"],
        status=DocumentStatus.UPLOADED,
        storage_path=saved["storage_path"],
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    background_tasks.add_task(process_document_by_id, document.id)

    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        original_filename=document.original_filename,
        status=document.status.value,
        message="Document uploaded successfully. Processing has started.",
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document)
        .where(
            Document.tenant_id == current_user.tenant_id,
            Document.status != DocumentStatus.DELETED,
        )
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", response_model=dict)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(status=DocumentStatus.DELETED)
    )
    await db.commit()

    from app.services.qdrant_service import delete_by_document_id
    delete_by_document_id(current_user.tenant_id, document_id)

    return {"message": "Document deleted successfully"}
