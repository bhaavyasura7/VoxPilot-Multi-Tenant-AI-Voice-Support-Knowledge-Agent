import json
import logging
import os
import tempfile

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.pipeline import retrieve_relevant_chunks
from app.services.agent_service import AgentSession
from app.redis_client import get_redis

logger = logging.getLogger(__name__)
settings = get_settings()

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

SESSION_TTL = 86400  # 24 hours


def _session_key(tenant_id: int, user_id: int) -> str:
    return f"session:voice:{tenant_id}:{user_id}"


async def _load_history(tenant_id: int, user_id: int) -> list[dict]:
    try:
        r = get_redis()
        data = await r.get(_session_key(tenant_id, user_id))
        if data:
            return json.loads(data)
    except Exception:
        pass
    return []


async def _save_history(tenant_id: int, user_id: int, history: list[dict]):
    try:
        r = get_redis()
        key = _session_key(tenant_id, user_id)
        await r.setex(key, SESSION_TTL, json.dumps(history))
    except Exception:
        pass


class VoiceSession:
    def __init__(self, tenant_id: int, user_id: int, org_name: str = "this organization"):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.org_name = org_name
        self.audio_buffer = bytearray()
        self.conversation_history: list[dict] = []

    @classmethod
    async def create(cls, tenant_id: int, user_id: int, org_name: str = "this organization"):
        session = cls(tenant_id, user_id, org_name)
        session.conversation_history = await _load_history(tenant_id, user_id)
        return session

    async def process_audio(self, audio_bytes: bytes) -> dict | None:
        self.audio_buffer.extend(audio_bytes)

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(self.audio_buffer)
                temp_path = f.name

            with open(temp_path, "rb") as audio_file:
                transcript = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            logger.info(f"Transcribed: '{transcript}' (tenant={self.tenant_id})")

            answer, sources = await self._process_query(str(transcript))

            self.conversation_history.append({"role": "user", "content": str(transcript)})
            self.conversation_history.append({"role": "assistant", "content": answer})
            await _save_history(self.tenant_id, self.user_id, self.conversation_history)

            self.audio_buffer.clear()

            return {
                "transcript": str(transcript),
                "answer": answer,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            self.audio_buffer.clear()
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _process_query(self, query: str) -> tuple[str, list[dict]]:
        results = await retrieve_relevant_chunks(
            query=query,
            tenant_id=self.tenant_id,
            top_k=4,
        )

        agent = AgentSession(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            org_name=self.org_name,
            voice_mode=True,
        )

        history_for_agent = self.conversation_history[-6:] if self.conversation_history else None
        answer, _ = await agent.run(query, history_for_agent)

        sources = [
            {"document_name": r["document_name"], "page_number": r["page_number"], "score": r["score"]}
            for r in results
        ]

        return answer, sources

    async def synthesize_speech(self, text: str) -> bytes:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3",
        )
        return response.content

    async def process_text(self, text: str) -> dict:
        answer, sources = await self._process_query(text)
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": answer})
        await _save_history(self.tenant_id, self.user_id, self.conversation_history)
        return {"answer": answer, "sources": sources}
