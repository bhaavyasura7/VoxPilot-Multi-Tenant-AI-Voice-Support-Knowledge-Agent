import json
import logging

from fastapi import APIRouter, WebSocket, Query
from fastapi.websockets import WebSocketState

from app.auth.jwt import decode_token
from app.services.voice_service import VoiceSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


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
    logger.info(f"Voice connected: user={user_id} tenant={tenant_id}")

    session = VoiceSession(tenant_id=tenant_id, user_id=int(user_id))

    async def safe_send(msg_type: str, payload: dict):
        try:
            await ws.send_json({**payload, "type": msg_type})
        except Exception:
            pass

    async def safe_send_bytes(data: bytes):
        try:
            await ws.send_bytes(data)
        except Exception:
            pass

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg:
                audio_bytes = msg["bytes"]

                if len(audio_bytes) < 1000:
                    await safe_send("warning", {"text": "Audio too short. Hold the mic button and speak clearly."})
                    continue

                result = await session.process_audio(audio_bytes)
                if result:
                    await safe_send("transcript", {"text": result["transcript"]})
                    await safe_send("answer", {"text": result["answer"], "sources": result["sources"]})

                    tts_audio = await session.synthesize_speech(result["answer"])
                    await safe_send_bytes(tts_audio)
                else:
                    await safe_send("error", {"text": "Could not process audio. Please try again."})

            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    action = data.get("action", "")

                    if action == "text_query":
                        query = data.get("query", "")
                        result = await session.process_text(query)
                        await safe_send("answer", {"text": result["answer"], "sources": result["sources"]})
                        tts_audio = await session.synthesize_speech(result["answer"])
                        await safe_send_bytes(tts_audio)

                except json.JSONDecodeError:
                    pass

    except Exception as e:
        logger.error(f"Voice disconnected: {e}")


@router.get("/config")
async def get_voice_config():
    return {
        "model": "whisper-1 + gpt-4o-mini + tts-1",
        "voice": "alloy",
        "sample_rate": 24000,
        "channels": 1,
        "format": "pcm16",
    }
