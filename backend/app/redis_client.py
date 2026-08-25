import redis.asyncio as redis
from app.config import get_settings

settings = get_settings()

_pool: redis.ConnectionPool | None = None


def get_redis() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=10,
            decode_responses=True,
        )
    return redis.Redis(connection_pool=_pool)


async def close_redis():
    global _pool
    if _pool:
        await _pool.disconnect()
        _pool = None


async def health_check() -> bool:
    try:
        r = get_redis()
        return await r.ping()
    except Exception:
        return False
