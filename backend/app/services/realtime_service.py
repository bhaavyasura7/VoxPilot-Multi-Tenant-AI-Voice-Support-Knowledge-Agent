import asyncio
import json
import logging
from typing import Callable, Awaitable

import websockets
from websockets.asyncio.client import ClientConnection

from app.config import get_settings
from app.tools.registry import registry
from app.tools.executor import execute_tool

# Import builtin tools to register them
import app.tools.builtin  # noqa: F401

logger = logging.getLogger(__name__)
settings = get_settings()

REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={settings.REALTIME_MODEL}"

VOICE_SYSTEM_PROMPT = """You are a helpful, friendly voice AI support agent for {org_name}.
Answer customer questions using your tools. Keep spoken answers concise — under 2-3 sentences.
Always search the knowledge base before answering factual questions.
Base answers ONLY on search results. Never invent policies or information.
If search returns nothing, say: "I couldn't find that in our knowledge base."
Be warm and conversational."""


class RealtimeSession:
    def __init__(
        self,
        tenant_id: int,
        user_id: int,
        org_name: str = "this organization",
        voice: str = "alloy",
        on_audio: Callable[[bytes], Awaitable[None]] | None = None,
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
        on_text_done: Callable[[str], Awaitable[None]] | None = None,
        on_transcript: Callable[[str], Awaitable[None]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, str], Awaitable[None]] | None = None,
        on_error: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.org_name = org_name
        self.voice = voice
        self._ws: ClientConnection | None = None
        self._openai_connected = False
        self._session_ready = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None
        self._speaking = False

        self.on_audio = on_audio or (lambda _: asyncio.sleep(0))
        self.on_text_delta = on_text_delta or (lambda _: asyncio.sleep(0))
        self.on_text_done = on_text_done or (lambda _: asyncio.sleep(0))
        self.on_transcript = on_transcript or (lambda _: asyncio.sleep(0))
        self.on_status = on_status or (lambda _: asyncio.sleep(0))
        self.on_tool_call = on_tool_call or (lambda _, __: asyncio.sleep(0))
        self.on_error = on_error or (lambda _: asyncio.sleep(0))

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, pcm16_bytes: bytes):
        if self._session_ready:
            await self._audio_queue.put(pcm16_bytes)

    async def send_text(self, text: str):
        if not self._ws or not self._session_ready:
            return
        msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await self._ws.send(json.dumps(msg))
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def _run(self):
        try:
            async with websockets.connect(
                REALTIME_URL,
                additional_headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                },
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                self._ws = ws
                self._openai_connected = True
                logger.info(f"Connected to OpenAI Realtime (tenant={self.tenant_id})")

                await self._configure_session()
                self._session_ready = True

                send_task = asyncio.create_task(self._audio_sender())
                recv_task = asyncio.create_task(self._event_receiver())

                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Realtime session error: {e}")
            await self.on_error(f"Connection error: {str(e)}")
        finally:
            self._openai_connected = False
            self._session_ready = False

    async def _configure_session(self):
        tools = _realtime_tools(registry.get_definitions())

        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": settings.REALTIME_MODEL,
                "output_modalities": ["audio"],
                "instructions": VOICE_SYSTEM_PROMPT.format(org_name=self.org_name),
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": settings.REALTIME_TRANSCRIPTION_MODEL,
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": self.voice,
                    },
                },
                "tools": tools,
                "tool_choice": "auto",
            },
        }
        await self._ws.send(json.dumps(session_config))
        logger.info(f"Session configured with {len(tools)} tools")

    async def _audio_sender(self):
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                msg = {
                    "type": "input_audio_buffer.append",
                    "audio": _bytes_to_b64(chunk),
                }
                await self._ws.send(json.dumps(msg))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio sender error: {e}")
                break

    async def _event_receiver(self):
        while self._running:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
                event = json.loads(raw)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("OpenAI Realtime connection closed")
                break
            except Exception as e:
                logger.error(f"Event receiver error: {e}")
                break

    async def _handle_event(self, event: dict):
        event_type = event.get("type", "")

        if event_type == "session.created":
            logger.info("OpenAI Realtime session created")

        elif event_type == "session.updated":
            logger.info("OpenAI Realtime session updated")

        elif event_type == "input_audio_buffer.speech_started":
            await self.on_status("listening")

        elif event_type == "input_audio_buffer.speech_stopped":
            await self.on_status("processing")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                await self.on_transcript(transcript)

        elif event_type == "response.output_audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64:
                if not self._speaking:
                    self._speaking = True
                    await self.on_status("speaking")
                await self.on_audio(_b64_to_bytes(audio_b64))

        elif event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                await self.on_text_delta(delta)

        elif event_type == "response.output_audio_transcript.delta":
            delta = event.get("delta", "")
            if delta:
                await self.on_text_delta(delta)

        elif event_type == "response.output_text.done":
            text = event.get("text", "")
            if text:
                await self.on_text_done(text)

        elif event_type == "response.output_audio_transcript.done":
            text = event.get("transcript", "")
            if text:
                await self.on_text_done(text)

        elif event_type == "response.done":
            self._speaking = False
            await self.on_status("idle")

        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            arguments = event.get("arguments", "{}")
            await self.on_tool_call(name, arguments)
            await self._execute_tool_and_respond(call_id, name, arguments)

        elif event_type == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            logger.error(f"OpenAI Realtime error: {error_msg}")
            await self.on_error(error_msg)

    async def _execute_tool_and_respond(self, call_id: str, name: str, arguments: str):
        try:
            result = await execute_tool(
                name=name,
                arguments=arguments,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
            )
        except Exception as e:
            result = json.dumps({"error": str(e)})

        item_msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self._ws.send(json.dumps(item_msg))
        await self._ws.send(json.dumps({"type": "response.create"}))


def _realtime_tools(definitions: list[dict]) -> list[dict]:
    tools = []
    for d in definitions:
        f = d.get("function", {})
        tools.append({
            "type": "function",
            "name": f.get("name"),
            "description": f.get("description"),
            "parameters": f.get("parameters"),
        })
    return tools


def _bytes_to_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def _b64_to_bytes(data: str) -> bytes:
    import base64
    return base64.b64decode(data)
