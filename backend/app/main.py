from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.database import init_db
from app.config import get_settings
from app.api import auth, organizations, documents, knowledge, chat, voice
from app.middleware.rate_limit import RateLimitMiddleware
from app.services.pubsub import start_listener, stop_listener
from app.redis_client import health_check as redis_health

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await start_listener()
    yield
    await stop_listener()


app = FastAPI(
    title="VoxPilot",
    description="Multi-Tenant AI Voice Support & Knowledge Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

app.include_router(auth.router)
app.include_router(organizations.router)
app.include_router(documents.router)
app.include_router(knowledge.router)
app.include_router(chat.router)
app.include_router(voice.router)

static_dir = Path(__file__).resolve().parent.parent / "static"


@app.get("/voice", response_class=HTMLResponse)
async def voice_ui():
    path = static_dir / "voice.html"
    return HTMLResponse(path.read_text() if path.exists() else "<h1>Not found</h1>")


@app.get("/webrtc", response_class=HTMLResponse)
async def webrtc_ui():
    path = static_dir / "webrtc.html"
    return HTMLResponse(path.read_text() if path.exists() else "<h1>Not found</h1>")


@app.get("/")
async def root():
    return {"message": "VoxPilot API is running"}


@app.get("/health")
async def health():
    redis_ok = await redis_health()
    return {"status": "healthy", "redis": redis_ok}
