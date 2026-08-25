from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_API_KEY: str = ""
    DATABASE_URL: str = "postgresql+asyncpg://voxpilot:voxpilot@db:5432/voxpilot"
    DATABASE_URL_SYNC: str = "postgresql://voxpilot:voxpilot@db:5432/voxpilot"
    REDIS_URL: str = "redis://redis:6379"
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 10
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    QDRANT_COLLECTION_NAME: str = "voxpilot_documents"
    RAG_TOP_K: int = 4
    RAG_SCORE_THRESHOLD: float = 0.5
    LLM_MODEL: str = "gpt-4o-mini"
    REALTIME_MODEL: str = "gpt-realtime"
    REALTIME_TRANSCRIPTION_MODEL: str = "whisper-1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
