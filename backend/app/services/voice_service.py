import asyncio
import json
import logging
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.pipeline import retrieve_relevant_chunks

logger = logging.getLogger(__name__)
settings = get_settings()

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

VOICE_SYSTEM_PROMPT = """You are a helpful, friendly voice AI support agent for Nimbus Home Goods Co.
Answer customer questions using ONLY the provided knowledge base context.

RULES:
- Keep spoken answers concise and natural — under 2-3 sentences.
- Base your answer ONLY on the provided context.
- If context doesn't answer the question, say: "I couldn't find that in our knowledge base."
- Never invent policies or information.
- Be warm and conversational."""


class VoiceSession:
    def __init__(self, tenant_id: int, user_id: int):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.audio_buffer = bytearray()
        self.conversation_history: list[dict] = []

    async def process_audio(self, audio_bytes: bytes) -> dict | None:
        """Transcribe audio using Whisper, then run RAG pipeline."""
        import tempfile, os

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

            answer, sources = await self._rag_query(str(transcript))

            self.conversation_history.append({"role": "user", "content": str(transcript)})
            self.conversation_history.append({"role": "assistant", "content": answer})

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

    async def _rag_query(self, query: str) -> tuple[str, list[dict]]:
        results = await retrieve_relevant_chunks(
            query=query,
            tenant_id=self.tenant_id,
            top_k=4,
        )

        context_text = "\n\n---\n\n".join(
            f"[{r['document_name']}, p{r['page_number']}]: {r['content'][:500]}"
            for r in results
        ) if results else "No relevant documents found."

        messages = [
            {"role": "system", "content": VOICE_SYSTEM_PROMPT},
        ]

        for msg in self.conversation_history[-6:]:
            messages.append(msg)

        messages.append({
            "role": "user",
            "content": f"Knowledge base context:\n{context_text}\n\nCustomer question: {query}",
        })

        response = await openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )

        answer = response.choices[0].message.content or "I couldn't generate a response."

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
            response_format="pcm",
        )
        return response.content

    async def process_text(self, text: str) -> dict:
        answer, sources = await self._rag_query(text)
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "sources": sources}
