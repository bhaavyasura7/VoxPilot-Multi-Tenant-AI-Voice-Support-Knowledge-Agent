import logging
from sqlalchemy import select

from app.database import async_session_factory
from app.models.conversation import Conversation, ConversationType, Message

logger = logging.getLogger(__name__)


async def create_voice_conversation(tenant_id: int, user_id: int, title: str | None = None) -> int:
    async with async_session_factory() as session:
        conv = Conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or "Voice Conversation",
            type=ConversationType.VOICE,
        )
        session.add(conv)
        await session.commit()
        await session.refresh(conv)
        return conv.id


async def save_voice_message(conversation_id: int, role: str, content: str) -> int:
    async with async_session_factory() as session:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        session.add(msg)
        await session.commit()
        return msg.id


async def update_conversation_title(conversation_id: int, title: str):
    async with async_session_factory() as session:
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = result.scalar_one_or_none()
        if conv:
            conv.title = title
            await session.commit()


async def get_or_create_conversation(
    tenant_id: int,
    user_id: int,
    conversation_id: int | None = None,
) -> int:
    if conversation_id:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.tenant_id == tenant_id,
                )
            )
            conv = result.scalar_one_or_none()
            if conv:
                return conv.id

    return await create_voice_conversation(tenant_id, user_id)
