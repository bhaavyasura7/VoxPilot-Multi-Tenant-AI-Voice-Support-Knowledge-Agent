from app.tools.registry import registry


async def get_conversation_history(
    user_id: int,
    conversation_id: int | None = None,
    limit: int = 5,
    **kwargs,
) -> str:
    from sqlalchemy import select
    from app.database import async_session_factory
    from app.models.conversation import Conversation, Message
    import json

    async with async_session_factory() as session:
        if conversation_id:
            result = await session.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            conversation = result.scalar_one_or_none()
            if not conversation:
                return json.dumps({"error": "Conversation not found."})

            msg_result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .limit(limit * 2)
            )
            messages = msg_result.scalars().all()
            history = [{"role": m.role, "content": m.content} for m in messages]
            return json.dumps({"conversation_id": conversation_id, "title": conversation.title, "messages": history})
        else:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            )
            conversations = result.scalars().all()
            items = [{"id": c.id, "title": c.title} for c in conversations]
            return json.dumps({"conversations": items, "total": len(items)})


registry.register(
    name="get_conversation_history",
    description="Retrieve past conversation history. Use this when the user asks about previous conversations, wants to reference something discussed earlier, or asks 'do you remember what we talked about?'. If no conversation_id is provided, lists recent conversations.",
    parameters={
        "type": "object",
        "properties": {
            "conversation_id": {
                "type": "integer",
                "description": "Optional. The ID of a specific conversation to fetch messages from. If omitted, lists recent conversations.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages or conversations to return. Default 5.",
                "default": 5,
            },
        },
        "required": [],
    },
    handler=get_conversation_history,
)
