from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.schemas.chat import ChatRequest, ConversationResponse, ChatMessageResponse
from app.auth.deps import get_current_user
from app.services.pipeline import retrieve_relevant_chunks
from app.services.openai_service import generate_rag_answer

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=dict)
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conversation = None
    if body.conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == body.conversation_id,
                Conversation.tenant_id == current_user.tenant_id,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation is None:
        term = body.message[:50] if len(body.message) > 50 else body.message
        conversation = Conversation(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            title=term,
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=body.message,
    )
    db.add(user_msg)
    await db.commit()

    results = await retrieve_relevant_chunks(
        query=body.message,
        tenant_id=current_user.tenant_id,
    )

    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
        .limit(10)
    )
    history_messages = history_result.scalars().all()
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_messages
        if msg.id != user_msg.id
    ]

    answer = await generate_rag_answer(
        query=body.message,
        context_chunks=results,
        conversation_history=conversation_history if conversation_history else None,
    )

    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.commit()

    sources = [
        {
            "document_name": r["document_name"],
            "page_number": r["page_number"],
            "score": r["score"],
        }
        for r in results
    ]

    return {
        "conversation_id": conversation.id,
        "answer": answer,
        "sources": sources,
    }


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.tenant_id == current_user.tenant_id,
            Conversation.user_id == current_user.id,
        )
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc())
    )
    return result.unique().scalars().all()


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == current_user.tenant_id,
        )
        .options(selectinload(Conversation.messages))
    )
    conversation = result.unique().scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation
