import asyncio
import logging
from typing import Callable, Awaitable

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCIceCandidate
from aiortc import AudioStreamTrack
from aiortc.mediastreams import AudioFrame

from app.services.realtime_service import RealtimeSession

logger = logging.getLogger(__name__)


def _parse_ice_candidate(candidate_str: str, sdp_mid: str, sdp_mline_index: int) -> RTCIceCandidate | None:
    if not candidate_str:
        return None
    prefix = "candidate:"
    if not candidate_str.startswith(prefix):
        return None
    parts = candidate_str[len(prefix):].split()
    if len(parts) < 8:
        return None

    try:
        # Format: foundation component protocol priority ip port typ type [raddr] [rport]
        typ_idx = parts.index("typ")
        candidate_type = parts[typ_idx + 1]
    except (ValueError, IndexError):
        return None

    return RTCIceCandidate(
        foundation=parts[0],
        component=int(parts[1]),
        protocol=parts[2].lower(),
        priority=int(parts[3]),
        ip=parts[4],
        port=int(parts[5]),
        type=candidate_type,
        sdpMid=sdp_mid,
        sdpMLineIndex=sdp_mline_index,
    )


class AudioRelayTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=100)
        self._sample_rate = sample_rate

    async def recv(self) -> AudioFrame:
        return await self._queue.get()

    def add_pcm16(self, pcm16_bytes: bytes):
        samples = len(pcm16_bytes) // 2
        if samples == 0:
            return
        frame = AudioFrame(
            channels=1,
            data=pcm16_bytes,
            sample_rate=self._sample_rate,
            samples=samples,
        )
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(frame)


class WebRTCSession:
    def __init__(
        self,
        tenant_id: int,
        user_id: int,
        org_name: str = "this organization",
        on_transcript: Callable[[str], Awaitable[None]] | None = None,
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
        on_text_done: Callable[[str], Awaitable[None]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, str], Awaitable[None]] | None = None,
        on_error: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.org_name = org_name

        self.on_transcript = on_transcript or (lambda _: asyncio.sleep(0))
        self.on_text_delta = on_text_delta or (lambda _: asyncio.sleep(0))
        self.on_text_done = on_text_done or (lambda _: asyncio.sleep(0))
        self.on_status = on_status or (lambda _: asyncio.sleep(0))
        self.on_tool_call = on_tool_call or (lambda _, __: asyncio.sleep(0))
        self.on_error = on_error or (lambda _: asyncio.sleep(0))
        self.on_ice_candidate: Callable[[dict], Awaitable[None]] | None = None

        self._pc: RTCPeerConnection | None = None
        self._realtime: RealtimeSession | None = None
        self._relay_track: AudioRelayTrack | None = None

    async def start(self):
        self._pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                ]
            )
        )

        self._realtime = RealtimeSession(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            org_name=self.org_name,
            on_audio=self._on_realtime_audio,
            on_text_delta=self.on_text_delta,
            on_text_done=self.on_text_done,
            on_transcript=self.on_transcript,
            on_status=self.on_status,
            on_tool_call=self.on_tool_call,
            on_error=self.on_error,
        )
        await self._realtime.start()

        @self._pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.info(f"WebRTC audio track from browser (tenant={self.tenant_id})")
                asyncio.create_task(self._process_audio_track(track))

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = self._pc.connectionState
            logger.info(f"WebRTC connection state: {state} (tenant={self.tenant_id})")
            if state in ("failed", "disconnected", "closed"):
                await self.on_error(f"WebRTC {state}")

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            state = self._pc.iceConnectionState
            logger.info(f"ICE connection state: {state} (tenant={self.tenant_id})")
            if state == "completed":
                logger.info(f"WebRTC ICE connected (tenant={self.tenant_id})")
                await self.on_status("listening")

        @self._pc.on("icecandidate")
        async def on_ice_candidate(event):
            if event.candidate and self.on_ice_candidate:
                await self.on_ice_candidate({
                    "candidate": event.candidate.candidate,
                    "sdpMid": event.candidate.sdpMid,
                    "sdpMLineIndex": event.candidate.sdpMLineIndex,
                })

    async def stop(self):
        if self._realtime:
            await self._realtime.stop()
            self._realtime = None
        if self._pc:
            await self._pc.close()
            self._pc = None

    async def handle_offer(self, sdp: str) -> str:
        if not self._pc:
            raise RuntimeError("WebRTC session not started")

        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self._pc.setRemoteDescription(offer)

        self._relay_track = AudioRelayTrack(sample_rate=24000)
        self._pc.addTrack(self._relay_track)

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        logger.info(f"WebRTC SDP answer ready (tenant={self.tenant_id})")
        return self._pc.localDescription.sdp

    async def handle_ice_candidate(self, candidate: str, sdp_mid: str, sdp_mline_index: int):
        ice = _parse_ice_candidate(candidate, sdp_mid, sdp_mline_index)
        if ice and self._pc:
            await self._pc.addIceCandidate(ice)

    async def send_text(self, text: str):
        if self._realtime:
            await self._realtime.send_text(text)

    async def _process_audio_track(self, track):
        try:
            while True:
                frame = await track.recv()
                if self._realtime:
                    await self._realtime.send_audio(frame.data)
        except Exception as e:
            logger.info(f"Browser audio track ended: {e}")

    async def _on_realtime_audio(self, pcm16_bytes: bytes):
        if self._relay_track:
            self._relay_track.add_pcm16(pcm16_bytes)
