from app.tools.registry import registry


async def list_documents(tenant_id: int, **kwargs) -> str:
    from sqlalchemy import select
    from app.database import async_session_factory
    from app.models.document import Document, DocumentStatus
    import json

    async with async_session_factory() as session:
        result = await session.execute(
            select(Document).where(
                Document.tenant_id == tenant_id,
                Document.status.in_([DocumentStatus.COMPLETED, DocumentStatus.PROCESSING]),
            ).order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()

    if not docs:
        return json.dumps({"message": "No documents available in the knowledge base."})

    items = []
    for d in docs:
        items.append({
            "id": d.id,
            "name": d.original_filename,
            "type": d.file_type,
            "status": d.status.value,
            "chunks": d.chunk_count,
        })

    return json.dumps({"documents": items, "total": len(items)})


registry.register(
    name="list_documents",
    description="List all available documents in the organization's knowledge base. Use this when the user asks what documents are available or what information you can help with.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    handler=list_documents,
)
