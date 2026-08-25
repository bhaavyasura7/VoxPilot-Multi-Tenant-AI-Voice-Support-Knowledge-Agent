import time
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

RATE_LIMIT_WINDOW = 60       # seconds
RATE_LIMIT_MAX_REQUESTS = 60  # per window
RATE_LIMIT_WHITELIST = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in RATE_LIMIT_WHITELIST:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        try:
            r = get_redis()
            key = f"ratelimit:{client_ip}"
            current = await r.get(key)

            if current is None:
                await r.setex(key, RATE_LIMIT_WINDOW, 1)
            else:
                count = int(current)
                if count >= RATE_LIMIT_MAX_REQUESTS:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
                await r.incr(key)

        except HTTPException:
            raise
        except Exception:
            pass

        return await call_next(request)


async def check_tenant_rate_limit(tenant_id: int, limit: int = 100, window: int = 60) -> bool:
    try:
        r = get_redis()
        key = f"ratelimit:tenant:{tenant_id}"
        current = await r.get(key)
        if current is None:
            await r.setex(key, window, 1)
            return True
        count = int(current)
        if count >= limit:
            return False
        await r.incr(key)
        return True
    except Exception:
        return True
