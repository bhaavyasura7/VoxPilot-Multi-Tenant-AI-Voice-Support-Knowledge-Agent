import hashlib
import json
import logging
from app.redis_client import get_redis

logger = logging.getLogger(__name__)
CACHE_TTL = 3600  # 1 hour


def _hash_query(tenant_id: int, query: str) -> str:
    raw = f"{tenant_id}:{query.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _hash_embedding(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


async def get_cached_rag(tenant_id: int, query: str) -> list[dict] | None:
    try:
        r = get_redis()
        key = f"cache:rag:{_hash_query(tenant_id, query)}"
        data = await r.get(key)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None


async def set_cached_rag(tenant_id: int, query: str, results: list[dict]):
    try:
        r = get_redis()
        key = f"cache:rag:{_hash_query(tenant_id, query)}"
        await r.setex(key, CACHE_TTL, json.dumps(results))
    except Exception:
        pass


async def invalidate_rag_cache(tenant_id: int):
    try:
        r = get_redis()
        pattern = f"cache:rag:*"
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor, match=pattern, count=100)
            for key in keys:
                await r.delete(key)
            if cursor == 0:
                break
    except Exception:
        pass


async def get_cached_embedding(text: str) -> list[float] | None:
    try:
        r = get_redis()
        key = f"cache:embed:{_hash_embedding(text)}"
        data = await r.get(key)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None


async def set_cached_embedding(text: str, embedding: list[float]):
    try:
        r = get_redis()
        key = f"cache:embed:{_hash_embedding(text)}"
        await r.setex(key, CACHE_TTL * 24, json.dumps(embedding))
    except Exception:
        pass
