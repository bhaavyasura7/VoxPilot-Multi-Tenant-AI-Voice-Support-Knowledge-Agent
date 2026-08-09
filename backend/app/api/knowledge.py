from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.chat import SearchRequest, SearchResponse, SearchResultItem, RagAnswerRequest
from app.auth.deps import get_current_user
from app.services.pipeline import retrieve_relevant_chunks
from app.services.openai_service import generate_rag_answer

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
):
    results = await retrieve_relevant_chunks(
        query=body.query,
        tenant_id=current_user.tenant_id,
    )

    return SearchResponse(
        results=[
            SearchResultItem(
                content=r["content"],
                document_name=r["document_name"],
                page_number=r["page_number"],
                score=r["score"],
            )
            for r in results
        ]
    )


@router.post("/ask", response_model=dict)
async def ask_knowledge(
    body: RagAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = await retrieve_relevant_chunks(
        query=body.query,
        tenant_id=current_user.tenant_id,
    )

    conversation_history = None
    if body.conversation_id:
        from app.models.conversation import Conversation, Message
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        conv_result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == body.conversation_id,
                Conversation.tenant_id == current_user.tenant_id,
            )
            .options(selectinload(Conversation.messages))
        )
        conv = conv_result.scalar_one_or_none()
        if conv and conv.messages:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in conv.messages[-10:]
            ]

    answer = await generate_rag_answer(
        query=body.query,
        context_chunks=results,
        conversation_history=conversation_history,
    )

    sources = [
        {
            "document_name": r["document_name"],
            "page_number": r["page_number"],
            "score": r["score"],
        }
        for r in results
    ]

    return {"answer": answer, "sources": sources}
