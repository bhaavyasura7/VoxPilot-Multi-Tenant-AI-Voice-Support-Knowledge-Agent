import asyncio
import logging

from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.models.document import Document, DocumentStatus
from app.services.document_service import extract_text_from_pdf, extract_text_from_docx
from app.services.chunking_service import chunk_pages
from app.services.embedding_service import generate_embeddings, generate_embedding
from app.services.qdrant_service import upsert_points, create_collection
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

EMBEDDING_BATCH_SIZE = 20


async def process_document_by_id(document_id: int) -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                logger.error(f"Document {document_id} not found")
                return

            await session.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(status=DocumentStatus.PROCESSING)
            )
            await session.commit()

            if document.file_type == "pdf":
                pages = extract_text_from_pdf(document.storage_path)
            elif document.file_type == "docx":
                pages = extract_text_from_docx(document.storage_path)
            else:
                raise ValueError(f"Unsupported file type: {document.file_type}")

            if not pages or all(not p["content"].strip() for p in pages):
                raise ValueError("No extractable text found in document.")

            chunks = chunk_pages(pages)

            if not chunks:
                raise ValueError("Document chunking produced no chunks.")

            create_collection()

            embeddings = []
            chunk_texts = [c["content"] for c in chunks]
            for i in range(0, len(chunk_texts), EMBEDDING_BATCH_SIZE):
                batch = chunk_texts[i : i + EMBEDDING_BATCH_SIZE]
                batch_embeddings = await generate_embeddings(batch)
                embeddings.extend(batch_embeddings)

            import uuid

            points = []
            for i, chunk in enumerate(chunks):
                points.append(
                    {
                        "id": str(uuid.uuid4()),
                        "vector": embeddings[i],
                        "payload": {
                            "tenant_id": str(document.tenant_id),
                            "document_id": str(document.id),
                            "document_name": document.original_filename,
                            "chunk_id": i,
                            "page_number": chunk["page_number"],
                            "content": chunk["content"],
                        },
                    }
                )

            upsert_points(points)

            await session.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(
                    status=DocumentStatus.COMPLETED,
                    chunk_count=len(chunks),
                )
            )
            await session.commit()

        logger.info(f"Document {document_id} processed successfully: {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Document processing failed for doc {document_id}: {e}")
        try:
            async with async_session() as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(
                        status=DocumentStatus.FAILED,
                        error_message=str(e),
                    )
                )
                await session.commit()
        except Exception as db_err:
            logger.error(f"Failed to update document status: {db_err}")

    finally:
        await engine.dispose()


async def retrieve_relevant_chunks(
    query: str,
    tenant_id: int,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    from app.services.qdrant_service import search as qdrant_search

    query_embedding = await generate_embedding(query)
    results = qdrant_search(
        query_vector=query_embedding,
        tenant_id=tenant_id,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return results


async def process_document_background(document_id: int) -> None:
    asyncio.create_task(process_document_by_id(document_id))
