import asyncio
import json
import logging
from typing import Callable, Awaitable
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "voxpilot:events"


EventCallback = Callable[[str, dict], Awaitable[None]]

_handlers: dict[str, list[EventCallback]] = {}
_listener_task: asyncio.Task | None = None


def subscribe(event_type: str, handler: EventCallback):
    if event_type not in _handlers:
        _handlers[event_type] = []
    _handlers[event_type].append(handler)


async def publish(event_type: str, payload: dict):
    try:
        r = get_redis()
        msg = json.dumps({"type": event_type, "payload": payload})
        await r.publish(CHANNEL_PREFIX, msg)
        logger.info(f"Published event: {event_type}")
    except Exception as e:
        logger.error(f"Publish error: {e}")


async def start_listener():
    global _listener_task
    if _listener_task:
        return

    _listener_task = asyncio.create_task(_listen())


async def stop_listener():
    global _listener_task
    if _listener_task:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None


async def _listen():
    while True:
        try:
            r = get_redis()
            pubsub = r.pubsub()
            await pubsub.subscribe(CHANNEL_PREFIX)

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    event_type = data.get("type", "")
                    payload = data.get("payload", {})
                    handlers = _handlers.get(event_type, [])
                    await asyncio.gather(*[h(event_type, payload) for h in handlers])
                except json.JSONDecodeError:
                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"PubSub listener error: {e}")
            await asyncio.sleep(5)
