from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Status = Literal["ok", "error"]


class RagSearchRequest(BaseModel):
    id: str = Field(..., description="Request id for logs/tracing")
    max_sources: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=0.8, ge=0.0, le=1.0)
    queries: list[str] = Field(..., min_length=1)


class RagChunk(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reference: Optional[str] = None
    text: str


class RagSearchResponse(BaseModel):
    id: str
    status: Status
    details: Optional[str] = None
    chunks: list[RagChunk] = []
