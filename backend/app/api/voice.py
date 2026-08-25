import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, WebSocket, Query
from fastapi.websockets import WebSocketState
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import decode_token
from app.auth.deps import get_current_user
from app.database import get_db
from app.config import get_settings
from app.models.user import User
from app.models.conversation import Conversation, ConversationType, Message
from app.services.realtime_service import RealtimeSession
from app.services.voice_persistence import (
    create_voice_conversation,
    save_voice_message,
    update_conversation_title,
)

logger = logging.getLogger(__name__)

settings = get_settings()

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


# ── Shared helper for wrapping callbacks with persistence ───────────────────

def _make_persistence_wrappers(tenant_id: int, user_id: int):
    state = {"conversation_id": None, "titled": False}

    async def on_connected():
        state["conversation_id"] = await create_voice_conversation(tenant_id, user_id)
        state["titled"] = False

    async def on_transcript(text: str):
        cid = state["conversation_id"]
        if cid:
            asyncio.create_task(save_voice_message(cid, "user", text))
            if not state["titled"]:
                title = text[:50].strip() if len(text) > 50 else text.strip()
                if title:
                    state["titled"] = True
                    asyncio.create_task(update_conversation_title(cid, title))

    async def on_text_done(text: str):
        cid = state["conversation_id"]
        if cid and text:
            asyncio.create_task(save_voice_message(cid, "assistant", text))

    return state, on_connected, on_transcript, on_text_done


# ── PCM16 Realtime endpoint ─────────────────────────────────────────────────

@router.websocket("/realtime")
async def voice_realtime(
    ws: WebSocket,
    token: str = Query(...),
):
    payload = decode_token(token)
    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")

    if not user_id or not tenant_id:
        await ws.close(code=4001)
        return

    await ws.accept()
    logger.info(f"Voice Realtime connected: user={user_id} tenant={tenant_id}")

    tid = int(tenant_id)
    uid = int(user_id)
    state, on_conn, save_transcript, save_response = _make_persistence_wrappers(tid, uid)
    await on_conn()

    async def _ws_send_json(data: dict):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(data)
        except Exception:
            pass

    async def _ws_send_bytes(data: bytes):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_bytes(data)
        except Exception:
            pass

    async def _send_transcript(text: str):
        await _ws_send_json({"type": "transcript", "text": text})
        await save_transcript(text)

    async def _send_response(text: str):
        await _ws_send_json({"type": "text_done", "text": text})
        await save_response(text)

    session = RealtimeSession(
        tenant_id=tid,
        user_id=uid,
        on_audio=lambda chunk: _ws_send_bytes(chunk),
        on_text_delta=lambda text: _ws_send_json({"type": "text_delta", "text": text}),
        on_text_done=_send_response,
        on_transcript=_send_transcript,
        on_status=lambda state: _ws_send_json({"type": "status", "state": state}),
        on_tool_call=lambda name, args: _ws_send_json({"type": "tool_call", "name": name}),
        on_error=lambda msg: _ws_send_json({"type": "error", "text": msg}),
    )

    await session.start()

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg:
                await session.send_audio(msg["bytes"])

            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "text_input":
                        text = data.get("text", "")
                        if text.strip():
                            await save_transcript(text)
                            await _ws_send_json({"type": "transcript", "text": text})
                            await session.send_text(text)
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        logger.error(f"Voice Realtime error: {e}")
    finally:
        await session.stop()
        logger.info(f"Voice Realtime disconnected: user={user_id}")


# ── WebRTC signaling endpoint ──────────────────────────────────────────────

@router.websocket("/webrtc")
async def voice_webrtc(
    ws: WebSocket,
    token: str = Query(...),
):
    payload = decode_token(token)
    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")

    if not user_id or not tenant_id:
        await ws.close(code=4001)
        return

    await ws.accept()
    logger.info(f"Voice WebRTC connected: user={user_id} tenant={tenant_id}")

    from app.services.webrtc_service import WebRTCSession

    tid = int(tenant_id)
    uid = int(user_id)
    state, on_conn, save_transcript, save_response = _make_persistence_wrappers(tid, uid)
    await on_conn()

    async def _ws_send_json(data: dict):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(data)
        except Exception:
            pass

    async def _send_transcript(text: str):
        await _ws_send_json({"type": "transcript", "text": text})
        await save_transcript(text)

    async def _send_response(text: str):
        await _ws_send_json({"type": "text_done", "text": text})
        await save_response(text)

    session = WebRTCSession(
        tenant_id=tid,
        user_id=uid,
        on_transcript=_send_transcript,
        on_text_delta=lambda text: _ws_send_json({"type": "text_delta", "text": text}),
        on_text_done=_send_response,
        on_status=lambda s: _ws_send_json({"type": "status", "state": s}),
        on_tool_call=lambda name, args: _ws_send_json({"type": "tool_call", "name": name}),
        on_error=lambda msg: _ws_send_json({"type": "error", "text": msg}),
    )
    session.on_ice_candidate = lambda ice: _ws_send_json({"type": "ice_candidate", **ice})
    await session.start()

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "offer":
                        sdp = data.get("sdp", "")
                        answer_sdp = await session.handle_offer(sdp)
                        await _ws_send_json({"type": "answer", "sdp": answer_sdp})

                    elif msg_type == "ice_candidate":
                        candidate = data.get("candidate", "")
                        sdp_mid = data.get("sdpMid", "")
                        sdp_mline_index = data.get("sdpMLineIndex", 0)
                        await session.handle_ice_candidate(candidate, sdp_mid, sdp_mline_index)

                    elif msg_type == "text_input":
                        text = data.get("text", "")
                        if text.strip():
                            await save_transcript(text)
                            await _ws_send_json({"type": "transcript", "text": text})
                            await session.send_text(text)

                except json.JSONDecodeError:
                    pass

    except Exception as e:
        logger.error(f"Voice WebRTC error: {e}")
    finally:
        await session.stop()
        logger.info(f"Voice WebRTC disconnected: user={user_id}")


# ── Voice conversation history REST API ─────────────────────────────────────

@router.get("/conversations", response_model=list[dict])
async def list_voice_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.tenant_id == current_user.tenant_id,
            Conversation.user_id == current_user.id,
            Conversation.type == ConversationType.VOICE,
        )
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.unique().scalars().all()

    return [
        {
            "id": c.id,
            "title": c.title,
            "type": c.type.value,
            "message_count": len(c.messages),
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "messages": [
                {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
                for m in c.messages
            ],
        }
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=dict)
async def get_voice_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == current_user.tenant_id,
            Conversation.type == ConversationType.VOICE,
        )
        .options(selectinload(Conversation.messages))
    )
    conv = result.unique().scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Voice conversation not found")

    return {
        "id": conv.id,
        "title": conv.title,
        "type": conv.type.value,
        "message_count": len(conv.messages),
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in conv.messages
        ],
    }


@router.get("/config")
async def get_voice_config():
    return {
        "model": settings.REALTIME_MODEL,
        "voice": "alloy",
        "audio_format": "pcm16 (24kHz, mono)",
        "features": ["streaming_audio", "server_vad", "function_calling", "webrtc", "conversation_persistence"],
    }
