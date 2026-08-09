from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str = Field(..., min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SearchResultItem(BaseModel):
    content: str
    document_name: str
    page_number: int
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class RagAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: int | None = None


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: int
    tenant_id: int
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse] = []

    model_config = {"from_attributes": True}
