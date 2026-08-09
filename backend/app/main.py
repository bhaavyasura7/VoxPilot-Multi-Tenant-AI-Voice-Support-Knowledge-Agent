from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.config import get_settings
from app.api import auth, organizations, documents, knowledge, chat, voice

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


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

app.include_router(auth.router)
app.include_router(organizations.router)
app.include_router(documents.router)
app.include_router(knowledge.router)
app.include_router(chat.router)
app.include_router(voice.router)

app.mount("/ui", StaticFiles(directory="../frontend/public", html=True), name="frontend")


@app.get("/")
async def root():
    return {"message": "VoxPilot API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
