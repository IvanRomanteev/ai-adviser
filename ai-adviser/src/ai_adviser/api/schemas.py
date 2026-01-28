# src/ai_adviser/api/schemas.py
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class EmbedResponse(BaseModel):
    dims: int
    embedding: list[float]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = Field(default=800, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    text: str


class RagChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=8, ge=1, le=50)
    # Optional identifiers for multi‑turn conversations. When supplied
    # responses will incorporate the previous messages associated with the
    # thread and the assistant will update the memory with the new
    # question/answer pair. If omitted the call is treated as stateless.
    thread_id: Optional[str] = None
    user_id: Optional[str] = None


class RagChunk(BaseModel):
    id: Optional[str] = None
    content: str
    source_file: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    raw: Optional[dict[str, Any]] = None


class RagChatResponse(BaseModel):
    answer: str
    chunks: list[RagChunk] = []
